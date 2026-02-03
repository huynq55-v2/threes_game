import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import pickle  # <--- THÊM VÀO: Để lưu file .pkl
import glob
from tqdm import tqdm # <--- THÊM VÀO: Để hiện thanh tiến trình

from gymnasium import spaces
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
import threes_rs  # Thư viện Rust của bạn
import multiprocessing  # <--- THƯ VIỆN ĐA LUỒNG

# Constants
SAVE_DIR = "./training_data"
os.makedirs(SAVE_DIR, exist_ok=True)
GAMMA = 0.99  # Hệ số giảm dần (quan trọng để tính quay lui)

# ==========================================
# PHẦN 1: ĐỊNH NGHĨA MÔI TRƯỜNG & MẠNG
# (Phải định nghĩa trước khi Load Model)
# ==========================================

class ThreesGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        # Khởi tạo game Rust
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
        
        # Scale reward (cho giống lúc train PPO)
        reward = reward * 0.01
        
        self.current_episode_reward += reward
        if len(next_hint_set) > 1:
            self.bonus_hint_count += 1

        masks = self.valid_action_mask()
        
        info = {
            "action_mask": masks,
            "score": self.current_episode_reward,
        }
        
        if done:
            max_val = max(next_board)
            # Dùng tqdm.write để không bị vỡ thanh process bar
            # tqdm.write(f"💀 Die! MaxTile: {int(max_val)} | Reward: {self.current_episode_reward:.2f}")
        
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
# PHẦN 2: LOGIC ĐA LUỒNG (SỬA ĐỔI)
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

# --- HÀM WORKER: CHẠY TRÊN TỪNG NHÂN CPU ---
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

    # File Text để ghi dữ liệu
    # Dùng mode 'a' (append) hoặc 'w'
    file_path = os.path.join(save_dir, f"data_worker_{worker_id}.txt")
    
    # Mở file sẵn để ghi liên tục (đỡ tốn RAM lưu list)
    with open(file_path, "w") as f:
        
        print(f"[Worker {worker_id}] Start generating {num_episodes} eps to text...")

        for ep in range(num_episodes):
            obs, _ = env.reset()
            done = False
            
            # Chỉ cần lưu Board (One-Hot) và Reward để tính toán
            ep_obs_board = []
            ep_rewards = []
            
            while not done:
                action_masks = env.get_wrapper_attr("valid_action_mask")()
                action, _ = model.predict(obs, action_masks=action_masks, deterministic=False)

                ep_obs_board.append(obs['board']) # Lưu One-Hot
                
                obs, reward, done, truncated, info = env.step(action)
                ep_rewards.append(reward)

            # Tính quay lui Gt
            discounted_returns = calculate_discounted_returns(ep_rewards, gamma)

            # --- GHI RA FILE TEXT NGAY LẬP TỨC ---
            # Format: "board_values | return_value"
            # Ví dụ: "0,0,1,2,3,6...|150.5"
            buffer_lines = []
            for i in range(len(ep_obs_board)):
                # 1. Giải mã One-Hot thành chuỗi số nguyên
                board_str = decode_one_hot_to_flat_string(ep_obs_board[i])
                
                # 2. Lấy giá trị Gt
                target_g = discounted_returns[i]
                
                # 3. Tạo dòng log
                line = f"{board_str}|{target_g:.4f}\n"
                buffer_lines.append(line)
            
            # Ghi cả ván vào file
            f.writelines(buffer_lines)
            f.flush() # Đẩy dữ liệu xuống ổ cứng ngay

    print(f"[Worker {worker_id}] Done. Saved text to {file_path}")
    return file_path

# --- HÀM MAIN CONTROLLER ---
def run_parallel_generation(model_path, total_episodes=1000, num_workers=4):
    print(f"=== Kích hoạt Multiprocessing: {num_workers} workers ===")
    print(f"Tổng số tập: {total_episodes} | Mỗi worker: {total_episodes // num_workers}")
    
    episodes_per_worker = total_episodes // num_workers
    
    # Tạo danh sách tham số cho từng worker
    worker_args = []
    for i in range(num_workers):
        worker_args.append((
            i,                  # worker_id
            model_path,         # path
            episodes_per_worker,# num_episodes
            GAMMA,              # gamma
            SAVE_DIR            # save dir
        ))

    # Khởi tạo Pool
    with multiprocessing.Pool(processes=num_workers) as pool:
        # starmap giúp truyền nhiều tham số vào hàm worker
        pool.starmap(worker_generate, worker_args)
    
    print("=== TẤT CẢ WORKER ĐÃ HOÀN THÀNH ===")

def decode_one_hot_to_flat_string(one_hot_board):
    """
    Chuyển One-Hot (16,4,4) thành chuỗi "0,0,3,6,12,..." cho điện thoại đọc.
    Logic ngược lại của _process_obs.
    """
    # 1. Tìm Rank (vị trí có giá trị 1 lớn nhất trên trục channel)
    # Shape: (4, 4)
    ranks = np.argmax(one_hot_board, axis=0) 
    
    flat_values = []
    # Duyệt từng ô để tính lại giá trị thực
    for r in range(4):
        for c in range(4):
            rank = ranks[r, c]
            val = 0
            if rank == 0: val = 0
            elif rank == 1: val = 1
            elif rank == 2: val = 2
            else:
                # Rank 3 -> Val 3
                # Rank 4 -> Val 6
                # Công thức: 3 * 2^(rank - 3)
                val = 3 * (2 ** (rank - 3))
            
            flat_values.append(str(int(val)))
            
    # Nối lại thành chuỗi: "1,2,3,6..."
    return ",".join(flat_values)

if __name__ == "__main__":
    # Để an toàn với PyTorch multiprocessing
    multiprocessing.set_start_method('spawn', force=True)

    # 1. Config
    CHECKPOINT_DIR = "./logs_ppo_threes_resnet/" 
    TOTAL_EPISODES = 2000  # Tổng số ván muốn gen
    
    # Số luồng = Số nhân CPU (hoặc số nhân - 1 để máy đỡ lag)
    NUM_WORKERS = os.cpu_count() 
    # Nếu muốn test thì để NUM_WORKERS = 2 thôi
    
    latest_model = get_latest_checkpoint(CHECKPOINT_DIR)
    
    if latest_model:
        # Chạy song song
        run_parallel_generation(latest_model, total_episodes=TOTAL_EPISODES, num_workers=NUM_WORKERS)
    else:
        print("Không tìm thấy model!")