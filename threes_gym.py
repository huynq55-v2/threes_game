import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sb3_contrib.common.wrappers import ActionMasker
import threes_rs  # Import thư viện Rust của bạn
from stable_baselines3.common.monitor import Monitor # <--- Import cái này

class ThreesGymEnv(gym.Env):
    """
    Custom Environment tuân thủ chuẩn Gymnasium cho game Threes!
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        # Khởi tạo game từ Rust (Giả sử bạn có class ThreesEnv đơn lẻ trong Rust binding)
        # Nếu Rust chỉ có VecEnv, bạn cần sửa Rust để expose struct Game đơn lẻ, 
        # hoặc dùng trick gọi VecEnv với num_envs=1.
        self.game = threes_rs.ThreesEnv() 
        
        # 1. Action Space: 4 hướng (0: Up, 1: Down, 2: Left, 3: Right)
        self.action_space = spaces.Discrete(4)
        
        # 2. Observation Space: Dùng Dict để chứa cả Board và Hint
        self.observation_space = spaces.Dict({
            # Board: 1 kênh, 4x4, giá trị từ 0-15 (Rank)
            "board": spaces.Box(low=0, high=15, shape=(1, 4, 4), dtype=np.float32),
            # Hint: Vector One-hot 13 phần tử
            "hint": spaces.Box(low=0, high=1, shape=(13,), dtype=np.float32),
        })
        
        # Mapping giá trị tile sang index (1->0, 2->1, 3->2, ...)
        self.TILE_MAP = {v: i for i, v in enumerate([1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072])}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Gọi Rust reset
        # YÊU CẦU: Rust binding phải trả về (board_state, hint_value)
        raw_board, raw_hint_set = self.game.reset()
        
        observation = self._process_obs(raw_board, raw_hint_set)
        info = {}
        return observation, info

    def step(self, action):
        # Gọi Rust step
        # YÊU CẦU: Rust binding phải trả về (next_board, reward, done, next_hint)
        # Nếu Rust trả về done dạng boolean, ta gán truncated = False
        next_board, reward, done, next_hint_set = self.game.step(int(action))
        
        # Xử lý Reward (Scale nhỏ lại để PPO học ổn định hơn)
        reward = reward * 0.1

        # --- THÊM ĐOẠN NÀY ĐỂ IN LOG RA MÀN HÌNH ---
        if done:
            # Lấy Max Tile từ bàn cờ
            # Lưu ý: next_board đang là list phẳng hoặc array
            max_val = max(next_board) 
            print(f"💀 Game Over! Reward: {reward:.2f} | MaxTile: {max_val}")
        # -------------------------------------------
        
        observation = self._process_obs(next_board, next_hint_set)
        truncated = False 
        info = {}
        
        return observation, reward, done, truncated, info

    def valid_action_mask(self):
        """
        Hàm quan trọng nhất cho Maskable PPO.
        Trả về mảng boolean [True, False, True, True] tương ứng với các action được phép.
        """
        # YÊU CẦU: Rust binding cần có hàm get_valid_moves() trả về list bool hoặc list int
        valid_moves = self.game.valid_moves() 
        
        # Nếu Rust trả về [0, 1, 0, 1] (int), ta convert sang bool
        return np.array(valid_moves, dtype=bool)

    def _process_obs(self, flat_board, hint_set):
        """Chuyển đổi dữ liệu thô từ Rust sang Tensor cho Neural Net"""
        
        # 1. Xử lý Board (Log2 Transform như cũ)
        board_np = np.array(flat_board, dtype=np.float32)
        ranks = np.zeros_like(board_np)
        ranks[board_np == 1] = 1
        ranks[board_np == 2] = 2
        mask = (board_np >= 3)
        ranks[mask] = np.floor(np.log2(board_np[mask] / 3.0) + 1e-5) + 3
        ranks = np.clip(ranks, 0, 15)
        
        # Reshape về (Channel, H, W) -> (1, 4, 4)
        board_final = ranks.reshape(1, 4, 4)
        
        # 2. Xử lý Hint (One-hot)
        hint_vec = np.zeros((13,), dtype=np.float32)
        # Giả sử hint_set là list các giá trị [1, 6]
        for h in hint_set:
            if h in self.TILE_MAP:
                hint_vec[self.TILE_MAP[h]] = 1.0
                
        return {
            "board": board_final,
            "hint": hint_vec
        }

# Hàm helper để tạo env (bắt buộc cho SubprocVecEnv)
def make_env():
    env = ThreesGymEnv()
    # 1. Action Masker (để chặn nước đi sai)
    env = ActionMasker(env, lambda env: env.valid_action_mask())
    
    # 2. Monitor (để Log Reward và Moves cho SB3)
    # allow_early_resets=True giúp tránh lỗi nếu reset game giữa chừng
    env = Monitor(env, allow_early_resets=True) 
    
    return env