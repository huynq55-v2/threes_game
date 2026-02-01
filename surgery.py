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
from stable_baselines3.common.monitor import Monitor
import threes_rs  # Thư viện Rust của bạn

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
        
        self.TILE_MAP = {v: i for i, v in enumerate([1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144])}

        self.current_episode_reward = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raw_board, raw_hint_set = self.game.reset()

        self.current_episode_reward = 0.0

        observation = self._process_obs(raw_board, raw_hint_set)
        return observation, {}

    def step(self, action):
        next_board, reward, done, next_hint_set = self.game.step(int(action))
        
        # --- THÊM ĐOẠN NÀY ĐỂ IN LOG RA MÀN HÌNH ---
        # THÊM DÒNG NÀY: Cộng dồn reward vào tổng
        self.current_episode_reward += reward
        
        # Sửa đoạn print
        if done:
            max_val = max(next_board)
            
            # In ra TỔNG REWARD (self.current_episode_reward) thay vì reward bước cuối
            print(f"💀 Die! MaxTile: {int(max_val)} | Total Reward: {self.current_episode_reward:.2f}")
        # -------------------------------------------

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

    def undo(self):
        # Gọi hàm undo từ Rust
        board_flat, reward, done, hint_set = self.game.undo()
        # Chuyển đổi sang observation format mà Model hiểu
        obs = self._process_obs(board_flat, hint_set)
        return obs

    def redo(self):
        board_flat, reward, done, hint_set = self.game.redo()
        obs = self._process_obs(board_flat, hint_set)
        return obs

# --- HÀM MAKE ENV (QUAN TRỌNG: Phải định nghĩa ở đây để multiprocessing gọi được) ---
def make_env():
    env = ThreesGymEnv()
    # 1. Action Masker
    env = ActionMasker(env, lambda env: env.valid_action_mask())
    # 2. Monitor: QUAN TRỌNG NHẤT ĐỂ HIỆN LOG SB3
    # Nó sẽ ghi lại Reward và Moves để hiển thị trong bảng log
    env = Monitor(env)
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
            nn.Linear(14, 64),
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

def surgery_on_checkpoint(old_path, new_path, env):
    # --- ĐỊNH NGHĨA LẠI POLICY_KWARGS CHO CHẮC CHẮN ---
    # (Phải trùng khớp với kiến trúc bác đã dùng để train 9.3M steps)
    policy_kwargs = dict(
        features_extractor_class=ThreesResNetExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=nn.GELU,
    )

    print("🏥 Bắt đầu ca phẫu thuật...")
    
    # 1. Khởi tạo model mới với kiến trúc 14 input (đã sửa trong Env của bác)
    new_model = MaskablePPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs, # Phải đảm bảo policy_kwargs đã update features_dim
        verbose=1
    )

    # 2. Load trọng số từ model cũ
    # Lưu ý: load với device="cpu" cho an toàn
    old_model = MaskablePPO.load(old_path, device="cpu")
    old_params = old_model.policy.state_dict()
    new_params = new_model.policy.state_dict()

    print("🧠 Đang chuyển giao ký ức...")
    for key in new_params.keys():
        if key in old_params:
            if new_params[key].shape == old_params[key].shape:
                # Nếu shape khớp (phần ResNet, phần Fusion), chép nguyên sang
                new_params[key].copy_(old_params[key])
            else:
                # Nếu lệch shape (chính là lớp Linear đầu tiên của Hint)
                print(f"✂️  Đang khâu vết mổ tại: {key}")
                old_weight = old_params[key] # Shape [64, 13]
                # Chép 13 cột cũ vào 13 cột đầu của model mới [64, 14]
                new_params[key][:, :13].copy_(old_weight)
                # Cột thứ 14 để mặc định (init là 0 hoặc random nhỏ)
        else:
            print(f"⚠️  Phát hiện vùng não mới: {key}")

    # 3. Cập nhật trọng số mới vào model mới
    new_model.policy.load_state_dict(new_params)
    
    # 4. Lưu lại bản "hồi sinh"
    new_model.save(new_path)
    print(f"✅ Phẫu thuật thành công! File mới đã sẵn sàng tại: {new_path}")

# --- THỰC THI ---
if __name__ == "__main__":
    # Nhớ khởi tạo env mới với shape 14 trước khi gọi hàm này
    test_env = make_env() 
    old_ckpt = "./logs_ppo_threes_resnet/ppo_resnet_9600000_steps.zip"
    new_ckpt = "./logs_ppo_threes_resnet/ppo_resnet_9600000_v2_14input.zip"
    
    surgery_on_checkpoint(old_ckpt, new_ckpt, test_env)