import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
import threes_rs  # Thư viện Rust của bạn

# --- CẤU HÌNH ---
NUM_CPU = 8             
TOTAL_TIMESTEPS = 20_000_000 
SAVE_DIR = "./logs_ppo_threes_resnet/"

# ==========================================
# PHẦN 1: MÔI TRƯỜNG (WRAPPER)
# ==========================================
class ThreesGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        # Khởi tạo game Rust
        self.game = threes_rs.ThreesEnv() 
        
        self.action_space = spaces.Discrete(4)
        
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=15, shape=(1, 4, 4), dtype=np.float32),
            "hint": spaces.Box(low=0, high=1, shape=(13,), dtype=np.float32),
        })
        
        self.TILE_MAP = {v: i for i, v in enumerate([1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072])}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raw_board, raw_hint_set = self.game.reset()
        observation = self._process_obs(raw_board, raw_hint_set)
        return observation, {}

    def step(self, action):
        next_board, reward, done, next_hint_set = self.game.step(int(action))
        
        # Scale reward
        reward = reward * 0.1 
        
        observation = self._process_obs(next_board, next_hint_set)
        return observation, reward, done, False, {}

    def valid_action_mask(self):
        valid_moves = self.game.valid_moves() 
        return np.array(valid_moves, dtype=bool)

    def _process_obs(self, flat_board, hint_set):
        # 1. Board
        board_np = np.array(flat_board, dtype=np.float32)
        ranks = np.zeros_like(board_np)
        ranks[board_np == 1] = 1
        ranks[board_np == 2] = 2
        mask = (board_np >= 3)
        ranks[mask] = np.floor(np.log2(board_np[mask] / 3.0) + 1e-5) + 3
        ranks = np.clip(ranks, 0, 15)
        board_final = ranks.reshape(1, 4, 4)
        
        # 2. Hint
        hint_vec = np.zeros((13,), dtype=np.float32)
        for h in hint_set:
            if h in self.TILE_MAP:
                hint_vec[self.TILE_MAP[h]] = 1.0
                
        return {"board": board_final, "hint": hint_vec}

# --- HÀM MAKE ENV (QUAN TRỌNG: Phải định nghĩa ở đây để multiprocessing gọi được) ---
def make_env():
    env = ThreesGymEnv()
    # Bọc ActionMasker để PPO nhìn thấy mask
    env = ActionMasker(env, lambda env: env.valid_action_mask())
    return env

# ==========================================
# PHẦN 2: MẠNG RESNET (FEATURE EXTRACTOR)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.gelu(out)
        return out

class ThreesResNetExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        # A. Board Branch
        self.embedding = nn.Embedding(16, 64)
        self.resnet = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        self.board_out_dim = 128 * 4 * 4
        
        # B. Hint Branch
        self.hint_net = nn.Sequential(
            nn.Linear(13, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU()
        )
        self.hint_out_dim = 64
        
        # C. Fusion
        combined_dim = self.board_out_dim + self.hint_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.GELU()
        )

    def forward(self, observations):
        board = observations["board"].long().squeeze(1).clamp(0, 15)
        hint = observations["hint"]
        
        # Board Path
        x = self.embedding(board)           
        x = x.permute(0, 3, 1, 2)           
        board_feat = self.resnet(x)         
        board_feat = board_feat.flatten(1)  
        
        # Hint Path
        hint_feat = self.hint_net(hint)     
        
        # Combine
        combined = torch.cat((board_feat, hint_feat), dim=1)
        return self.fusion(combined)

# ==========================================
# PHẦN 3: MAIN LOOP
# ==========================================
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"🚀 Đang khởi tạo {NUM_CPU} môi trường song song...")
    vec_env = SubprocVecEnv([make_env for _ in range(NUM_CPU)])

    policy_kwargs = dict(
        features_extractor_class=ThreesResNetExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=nn.GELU,
    )

    print("🔥 Khởi tạo Maskable PPO với ResNet Backbone...")
    model = MaskablePPO(
        "MultiInputPolicy",
        vec_env,
        learning_rate=2.5e-4,   
        n_steps=2048,           
        batch_size=1024,        
        n_epochs=10,            
        gamma=0.99,             
        gae_lambda=0.95,        
        clip_range=0.2,
        ent_coef=0.01,          
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tensorboard_threes/",
        verbose=1,
        device="cpu"            
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000, 
        save_path=SAVE_DIR,
        name_prefix="ppo_resnet"
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
    model.save("threes_resnet_final")
    print("✅ Training Hoàn tất!")