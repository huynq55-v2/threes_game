import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import pickle
import glob
from tqdm import tqdm
import multiprocessing
import shutil  # <--- THÊM: Để ghép file
from datetime import datetime # <--- THÊM: Để lấy timestamp

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
import threes_rs

# Constants
SAVE_DIR = "./training_data"
os.makedirs(SAVE_DIR, exist_ok=True)
GAMMA = 0.99

# ==========================================
# PHẦN 1: ĐỊNH NGHĨA MÔI TRƯỜNG & MẠNG
# ==========================================

class ThreesGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.game = threes_rs.ThreesEnv()
        self.game.set_gamma(GAMMA)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(16, 4, 4), dtype=np.float32),
            "hint": spaces.Box(low=0, high=1, shape=(14,), dtype=np.float32),
        })
        self.TILE_MAP = {v: i for i, v in enumerate([1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144])}
        self.current_episode_reward = 0.0
        self.bonus_hint_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        raw_board, raw_hint_set = self.game.reset()
        self.current_episode_reward = 0.0
        observation = self._process_obs(raw_board, raw_hint_set)
        self.bonus_hint_count = 0
        return observation, {}

    def step(self, action):
        next_board, reward, done, next_hint_set = self.game.step(int(action))
        reward = reward * 0.01
        self.current_episode_reward += reward
        if len(next_hint_set) > 1:
            self.bonus_hint_count += 1
        masks = self.valid_action_mask()
        info = {
            "action_mask": masks,
            "score": self.current_episode_reward,
        }
        observation = self._process_obs(next_board, next_hint_set)
        return observation, reward, done, False, info

    def valid_action_mask(self):
        valid_moves = self.game.valid_moves() 
        return np.array(valid_moves, dtype=bool)

    def _process_obs(self, flat_board, hint_set):
        board_np = np.array(flat_board, dtype=np.float32)
        ranks = np.zeros_like(board_np)
        ranks[board_np == 1] = 1
        ranks[board_np == 2] = 2
        mask = (board_np >= 3)
        ranks[mask] = np.floor(np.log2(board_np[mask] / 3.0) + 1e-5) + 3
        ranks_int = ranks.astype(int).reshape(4, 4)
        one_hot = np.zeros((16, 4, 4), dtype=np.float32)
        for r in range(16):
            one_hot[r, :, :] = (ranks_int == r).astype(np.float32)
        hint_vec = np.zeros((14,), dtype=np.float32)
        for h in hint_set:
            if h in self.TILE_MAP:
                hint_vec[self.TILE_MAP[h]] = 1.0
        return {"board": one_hot, "hint": hint_vec}

def make_env():
    env = ThreesGymEnv()
    env = ActionMasker(env, lambda env: env.valid_action_mask())
    env = Monitor(env)
    return env

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
        self.resnet = nn.Sequential(
            nn.Conv2d(16, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        self.board_out_dim = 128 * 4 * 4
        self.hint_net = nn.Sequential(
            nn.Linear(14, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU()
        )
        self.hint_out_dim = 64
        combined_dim = self.board_out_dim + self.hint_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.GELU()
        )

    def forward(self, observations):
        board = observations["board"].float()
        hint = observations["hint"]
        board_feat = self.resnet(board)         
        board_feat = board_feat.flatten(1)  
        hint_feat = self.hint_net(hint)     
        combined = torch.cat((board_feat, hint_feat), dim=1)
        return self.fusion(combined)

# ==========================================
# PHẦN 2: LOGIC GEN DATA & MERGE
# ==========================================

def calculate_discounted_returns(rewards, gamma):
    returns = []
    g = 0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.insert(0, g)
    return returns

def get_latest_checkpoint(checkpoint_dir):
    list_of_files = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def decode_one_hot_to_flat_string(one_hot_board):
    ranks = np.argmax(one_hot_board, axis=0) 
    flat_values = []
    for r in range(4):
        for c in range(4):
            rank = ranks[r, c]
            val = 0
            if rank == 0: val = 0
            elif rank == 1: val = 1
            elif rank == 2: val = 2
            else:
                val = 3 * (2 ** (rank - 3))
            flat_values.append(str(int(val)))
    return ",".join(flat_values)

# --- WORKER FUNCTION ---
def worker_generate(worker_id, model_path, num_episodes, gamma, save_dir):
    torch.set_num_threads(1)
    env = make_env()
    
    try:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
        model = MaskablePPO.load(model_path, env=env, device='cpu', custom_objects=custom_objects)
    except:
        model = MaskablePPO.load(model_path, device='cpu')

    # File tạm cho worker
    file_path = os.path.join(save_dir, f"temp_worker_{worker_id}.txt")
    
    with open(file_path, "w") as f:
        print(f"[Worker {worker_id}] Start generating {num_episodes} eps...")
        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            ep_obs_board = []
            ep_rewards = []
            ep_actions = []
            
            while not done:
                action_masks = env.get_wrapper_attr("valid_action_mask")()
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=False)
                ep_actions.append(action)
                ep_obs_board.append(obs['board'])
                obs, reward, done, truncated, info = env.step(action)
                ep_rewards.append(reward)

            discounted_returns = calculate_discounted_returns(ep_rewards, gamma)

            buffer_lines = []
            for i in range(len(ep_obs_board)):
                board_str = decode_one_hot_to_flat_string(ep_obs_board[i])
                target_g = discounted_returns[i]
                
                # --- THÊM CÁI NÀY: Lấy Action sư phụ đã chọn ---
                action_taken = ep_actions[i] 
                
                # Format mới: "Board | G_Value | Action"
                # Ví dụ: "1,2,3... | 150.5 | 0" (0 là UP)
                line = f"{board_str}|{target_g:.4f}|{int(action_taken)}\n"
                
                buffer_lines.append(line)
            
            f.writelines(buffer_lines)
            f.flush()

    print(f"[Worker {worker_id}] Finished.")
    return file_path

# --- MAIN CONTROLLER (Đã sửa logic Merge) ---
def run_parallel_generation(model_path, total_episodes=1000, num_workers=4):
    print(f"=== Kích hoạt Multiprocessing: {num_workers} workers ===")
    print(f"Tổng số tập: {total_episodes} | Mỗi worker: {total_episodes // num_workers}")
    
    episodes_per_worker = total_episodes // num_workers
    worker_args = []
    for i in range(num_workers):
        worker_args.append((i, model_path, episodes_per_worker, GAMMA, SAVE_DIR))

    # 1. Chạy song song
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Nhận lại danh sách file tạm từ các worker
        temp_files = pool.starmap(worker_generate, worker_args)
    
    # 2. Tạo tên file hợp nhất với TimeStamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_filename = f"threes_ppo_data_{timestamp}.txt"
    final_path = os.path.join(SAVE_DIR, final_filename)

    print(f"=== Đang hợp nhất {len(temp_files)} files thành: {final_filename} ===")
    
    # 3. Logic Merge (Ghép file siêu tốc)
    with open(final_path, 'wb') as outfile: # Mở chế độ binary để copy cho nhanh
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                with open(temp_file, 'rb') as infile:
                    # Copy toàn bộ nội dung file con sang file tổng
                    shutil.copyfileobj(infile, outfile)
                
                # 4. Xóa file tạm sau khi ghép xong
                os.remove(temp_file)
                print(f"Merged & Deleted: {temp_file}")

    print(f"\n=== HOÀN TẤT! File dữ liệu tại: {final_path} ===")

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    CHECKPOINT_DIR = "./logs_ppo_threes_resnet/" 
    TOTAL_EPISODES = 2000 
    NUM_WORKERS = os.cpu_count()
    
    latest_model = get_latest_checkpoint(CHECKPOINT_DIR)
    
    if latest_model:
        run_parallel_generation(latest_model, total_episodes=TOTAL_EPISODES, num_workers=NUM_WORKERS)
    else:
        print("Không tìm thấy model!")