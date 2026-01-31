import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math
import os
import time
from collections import deque
import threes_rs  # Rust binding

# --- CẤU HÌNH (CONFIGURATION) ---
NUM_ENVS = 128          # Số lượng môi trường chạy song song
BATCH_SIZE = 256        
GAMMA = 0.98           # Increased for longer horizon
N_STEPS = 5             # N-Step Learning
LR = 1e-4               
TARGET_UPDATE = 5000    
MEMORY_SIZE = 500000    

# --- CHECKPOINT PATHS ---
BASE_DIR = "/kaggle/input/threes-binary"

def find_latest_checkpoint(base_dir, filename="threes_nstep_dueling_noisy_latest.pth"):
    versions = []

    # Check if base dir exists first
    if not os.path.exists(base_dir):
        return None

    # duyệt toàn bộ cây thư mục
    for root, dirs, files in os.walk(base_dir):
        if filename in files:
            # lấy version từ path: .../default/<version>/
            parts = root.split(os.sep)
            for p in reversed(parts):
                if p.isdigit():
                    versions.append((int(p), os.path.join(root, filename)))
                    break

    if not versions:
        return None

    # lấy version lớn nhất
    versions.sort(key=lambda x: x[0], reverse=True)
    return versions[0][1]

# Try to find existing checkpoint
CHECKPOINT_FILE = find_latest_checkpoint(BASE_DIR)
# Always save to working dir
CHECKPOINT_SAVE = "/kaggle/working/threes_nstep_dueling_noisy_latest.pth"

if CHECKPOINT_FILE:
    print(f"🔄 Found checkpoint to load: {CHECKPOINT_FILE}")
else:
    print(f"⚠️ No existing checkpoint found. Will initialize new model.")


# Mapping 13 loại quân (1, ... 3072) -> Index (0..12)
TILE_TYPES = [1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072]
TILE_MAP = {v: i for i, v in enumerate(TILE_TYPES)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 1. NOISY LINEAR LAYER ---
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters (Mu & Sigma)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Factorized Gaussian Noise
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            # W = mu + sigma * epsilon
            return F.linear(input, 
                            self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            # Evaluation mode: Use mean only (no noise)
            return F.linear(input, self.weight_mu, self.bias_mu)

# --- 2. RESNET BACKBONE & DUELING NOISY DQN ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # Using GELU as recommended for better gradient flow
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip Connection
        out = F.gelu(out)
        return out

class DuelingNoisyDQN(nn.Module):
    def __init__(self, num_blocks=2):
        super(DuelingNoisyDQN, self).__init__()
        
        # 1. Broad Vision (Board)
        self.embedding = nn.Embedding(16, 64) 
        
        # ResNet Backbone (Input 64 -> Output 256)
        # Keeps 4x4 spatial dim
        self.resnet_backbone = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            *[ResidualBlock(256) for _ in range(num_blocks)]
        )
        
        # ResNet Output: 256 channels * 4 * 4 = 4096
        self.board_feat_dim = 256 * 4 * 4
        
        # 2. Focused Vision (Hint)
        # Project 13-dim hints to 128-dim to avoid signal dilution
        self.hint_net = nn.Sequential(
            nn.Linear(13, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU()
        )
        
        # 3. Fusion
        self.combined_dim = self.board_feat_dim + 128
        
        # Advantage Stream (Noisy)
        self.advantage_stream = nn.Sequential(
            NoisyLinear(self.combined_dim, 1024),
            nn.GELU(),
            NoisyLinear(1024, 4)
        )
        
        # Value Stream (Noisy)
        self.value_stream = nn.Sequential(
            NoisyLinear(self.combined_dim, 1024),
            nn.GELU(),
            NoisyLinear(1024, 1)
        )
        
        # Weight Initialization usually helps ResNet
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, state):
        board = state[:, :16].long()
        hints = state[:, 16:].float()
        
        # --- BOARD PATH ---
        x = self.embedding(board.clamp(0, 15))
        x = x.view(-1, 4, 4, 64) 
        # Important: Permute (Batch, H, W, C) -> (Batch, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        board_features = self.resnet_backbone(x)
        board_features = board_features.reshape(board_features.size(0), -1) # Flatten
        
        # --- HINT PATH ---
        hint_features = self.hint_net(hints)
        
        # --- FUSION ---
        features = torch.cat((board_features, hint_features), dim=1)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Dueling Aggregation
        return values + (advantages - advantages.mean(dim=1, keepdim=True))
    
    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

# --- HELPER CLASSES (Augmentation & Replay) ---
class DataAugmenter:
    def __init__(self):
        # Move permutations to device immediately
        self.BOARD_PERMS = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], # Id
            [12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3], # Rot90
            [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], # Rot180
            [3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12], # Rot270
            [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12], # FlipX
            [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15], # FlipMainDiag
            [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3], # FlipY
            [15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0], # FlipAntiDiag
        ], dtype=torch.long, device=device)
        
        # Pre-calculate combined perms for actions to be safe and clear
        self.ACTION_PERMS = torch.tensor([
            [0, 1, 2, 3], # Id
            [3, 2, 0, 1], # Rot90
            [1, 0, 3, 2], # Rot180
            [2, 3, 1, 0], # Rot270
            [0, 1, 3, 2], # FlipX 
            [3, 2, 1, 0], # Rot90(FlipX)
            [1, 0, 2, 3], # Rot180(FlipX)
            [2, 3, 0, 1], # Rot270(FlipX)
        ], dtype=torch.long, device=device)

    def augment_tensors(self, states, actions, next_states):
        # randomly select one symmetry for the WHOLE batch for speed
        idx = random.randint(0, 7)
        if idx == 0:
            return states, actions, next_states
            
        perm_b = self.BOARD_PERMS[idx]
        perm_a = self.ACTION_PERMS[idx]
        
        # States: [Board (16) | Hints (13)]
        boards = states[:, :16]
        hints = states[:, 16:]
        
        # GPU Indexing: automatic broadcasting
        b_sym = boards[:, perm_b]
        states_aug = torch.cat([b_sym, hints], dim=1)
        
        # Next States
        nboards = next_states[:, :16]
        nhints = next_states[:, 16:]
        nb_sym = nboards[:, perm_b]
        next_states_aug = torch.cat([nb_sym, nhints], dim=1)
        
        # Actions: perm_a[actions]
        # actions must be LongTensor
        actions_aug = perm_a[actions]
        
        return states_aug, actions_aug, next_states_aug

class AdaptiveGammaScheduler:
    def __init__(self, init_gamma=0.95, min_gamma=0.90, max_gamma=0.995, horizon_ratio=1.75):
        self.gamma = init_gamma
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.ratio = horizon_ratio
        
        # Biến để theo dõi năng lực trung bình (EMA)
        # Khởi tạo giả định là 50 moves
        self.avg_moves_ema = 50.0 
        self.alpha = 0.05 # Hệ số làm mượt (càng nhỏ càng ổn định)

    def step(self, recent_moves_list):
        """
        Gọi hàm này sau mỗi Episode hoặc mỗi Batch log
        recent_moves_list: List các moves vừa chơi xong (ví dụ batch 128 envs)
        """
        if not recent_moves_list:
            return self.gamma

        # 1. Cập nhật năng lực trung bình (EMA)
        current_batch_avg = np.mean(recent_moves_list)
        self.avg_moves_ema = (1 - self.alpha) * self.avg_moves_ema + self.alpha * current_batch_avg
        
        # 2. Tính Horizon mục tiêu (Tầm nhìn = 1.75 lần Tuổi thọ)
        target_horizon = self.avg_moves_ema * self.ratio
        
        # Tránh chia cho 0
        target_horizon = max(target_horizon, 10.0)
        
        # 3. Tính Gamma từ Horizon: H = 1 / (1 - gamma) => gamma = 1 - 1/H
        new_gamma = 1.0 - (1.0 / target_horizon)
        
        # 4. Kẹp giá trị (Safety Clip)
        # Quan trọng: Chỉ cho phép Gamma tăng hoặc giữ nguyên, KHÔNG GIẢM.
        # Nếu giảm nghĩa là AI đang chơi ngu đi, ta không nên giảm tầm nhìn mà phải giữ nguyên để nó vượt qua.
        self.gamma = np.clip(new_gamma, self.gamma, self.max_gamma)
        
        return self.gamma

    def get_gamma(self):
        return self.gamma

def prepare_state_batch(boards_flat, hint_sets):
    """
    Chuyển đổi Raw Values (0, 1, 2, 3, 6, 12...) thành Embedding Indices (0..15).
    Đồng thời tạo Multi-hot vector cho Hints.
    """
    # 1. Xử lý Board: Value -> Rank (Vectorized)
    # Chuyển sang float để tính log2, sau đó ép về int
    boards_np = np.array(boards_flat, dtype=np.float32)
    
    # Khởi tạo mảng ranks toàn số 0 (tương ứng với ô trống)
    ranks = np.zeros_like(boards_np)
    
    # Case đặc biệt: 1 và 2
    # Đây là điểm quan trọng nhất để AI phân biệt được màu xanh/đỏ
    ranks[boards_np == 1] = 1
    ranks[boards_np == 2] = 2
    
    # Case tổng quát: Các số >= 3
    # Công thức: log2(value / 3) + 3
    # Ví dụ: 3 -> log2(1) + 3 = 3
    #        6 -> log2(2) + 3 = 4
    #        48 -> log2(16) + 3 = 7
    mask = (boards_np >= 3)
    # Thêm epsilon nhỏ hoặc dùng hàm round để tránh lỗi floating point (vd: 2.999 -> 2)
    ranks[mask] = np.floor(np.log2(boards_np[mask] / 3.0) + 1e-5) + 3
    
    # Clip lại để đảm bảo không vượt quá kích thước Embedding (0..15)
    # 15 tương ứng với số 12,288 (rất khó đạt được)
    ranks = np.clip(ranks, 0, 15)
    
    # 2. Xử lý Hints: Multi-hot Encoding
    n = len(boards_flat)
    multi_hots = np.zeros((n, 13), dtype=np.float32)
    
    # Hint sets thường nhỏ (1-3 phần tử), loop Python ở đây không ảnh hưởng nhiều hiệu năng
    # TILE_MAP đã định nghĩa ở ngoài: {1:0, 2:1, 3:2, 6:3...}
    for i, hints in enumerate(hint_sets):
        for h in hints:
            if h in TILE_MAP:
                multi_hots[i, TILE_MAP[h]] = 1.0
                
    # Nối lại: Board Ranks (16 ô) + Hint Vector (13 ô)
    return np.concatenate([ranks, multi_hots], axis=1)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push_batch(self, states, actions, rewards, next_states, dones):
        self.buffer.extend(zip(states, actions, rewards, next_states, dones))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (torch.tensor(np.array(s), dtype=torch.float32, device=device),
                torch.tensor(a, dtype=torch.long, device=device),
                torch.tensor(r, dtype=torch.float32, device=device),
                torch.tensor(np.array(ns), dtype=torch.float32, device=device),
                torch.tensor(d, dtype=torch.float32, device=device))

class NStepBuffer:
    def __init__(self, num_envs, n_step, gamma):
        self.num_envs = num_envs
        self.n_step = n_step
        self.gamma = gamma
        self.buffers = [deque(maxlen=n_step) for _ in range(num_envs)]
        
    def add(self, states, actions, rewards, next_states, dones):
        """
        states, actions, ..., dones: arrays needed (size=NUM_ENVS)
        Returns: list of valid transitions (s, a, R, ns, d) ready for ReplayBuffer
        """
        transitions_to_store = []
        
        for i in range(self.num_envs):
            # Add to env-specific buffer
            self.buffers[i].append((states[i], actions[i], rewards[i], next_states[i], dones[i]))
            
            # If done or full, compute return
            if dones[i]:
                # Flush buffer logic: for all stored items, compute return
                # However, classic N-step usually is:
                # If d: T is terminal.
                # Calc returns for all t in buffer.
                while len(self.buffers[i]) > 0:
                    s, a, _, _, _ = self.buffers[i][0]
                    # Calculate R
                    R = 0
                    for k, item in enumerate(self.buffers[i]):
                         r_k = item[2]
                         R += (self.gamma ** k) * r_k
                    
                    # Next state is terminal (next_states of last item is terminal)
                    # We store the terminal next_state of the LAST item in the buffer
                    ns_end = self.buffers[i][-1][3]
                    
                    transitions_to_store.append((s, a, R, ns_end, True))
                    self.buffers[i].popleft()
                    
            elif len(self.buffers[i]) == self.n_step:
                # Standard N-step update
                s, a, _, _, _ = self.buffers[i][0]
                R = 0
                for k, item in enumerate(self.buffers[i]):
                     r_k = item[2]
                     R += (self.gamma ** k) * r_k
                
                # Next state is state n steps later (or next_state of last item)
                ns_n = self.buffers[i][-1][3]
                
                transitions_to_store.append((s, a, R, ns_n, False))
                self.buffers[i].popleft()
                
        return transitions_to_store

def save_checkpoint(episode, steps, policy_net, optimizer):
    checkpoint = {
        'episode': episode,
        'steps': steps,
        'model_state': policy_net.state_dict(),
        'optimizer_state': optimizer.state_dict(),
    }
    torch.save(checkpoint, CHECKPOINT_SAVE)
    print(f"💾 Checkpoint saved to {CHECKPOINT_SAVE} at ep {episode}")

def load_checkpoint(policy_net, optimizer, target_net):
    load_path = CHECKPOINT_FILE
    
    if load_path and os.path.exists(load_path):
        print(f"🔄 Loading checkpoint from: {load_path}")
        try:
            checkpoint = torch.load(load_path, map_location=device, weights_only=False)
            policy_net.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            target_net.load_state_dict(policy_net.state_dict())
            return checkpoint['episode'], checkpoint['steps']
        except Exception as e:
            print(f"⚠️ Error loading checkpoint (Architecture mismatch?): {e}")
            print("⚠️ Starting from scratch since architecture changed!")
            return 0, 0
            
    print("⚠️ No checkpoint found. Starting from scratch.")
    return 0, 0

# --- MAIN LOOP ---
if __name__ == "__main__":
    # Init Objects (Using DuelingNoisyDQN)
    policy_net = DuelingNoisyDQN().to(device)
    target_net = DuelingNoisyDQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    
    # IMPORTANT: Target Net is set to eval() to disable noise sampling during target calculation
    # (Or keep it in train mode if we want noise in target? usually eval is safer for stability)
    target_net.eval() 

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)
    n_step_buffer = NStepBuffer(NUM_ENVS, N_STEPS, GAMMA)
    criterion = nn.HuberLoss()
    augmenter = DataAugmenter()
    
    start_episode, total_steps = load_checkpoint(policy_net, optimizer, target_net)
    
    vec_env = threes_rs.ThreesVecEnv(NUM_ENVS)
    
    WINDOW_SIZE = 100
    metrics = {
        'r': deque(maxlen=WINDOW_SIZE), 
        'm': deque(maxlen=WINDOW_SIZE), 
        'l': deque(maxlen=WINDOW_SIZE), 
        'q': deque(maxlen=WINDOW_SIZE), 
        't': deque(maxlen=WINDOW_SIZE) 
    }
    
    env_curr = {
        'moves': np.zeros(NUM_ENVS),
        'rewards': np.zeros(NUM_ENVS),
        'max_tile': np.zeros(NUM_ENVS)
    }

    gamma_scheduler = AdaptiveGammaScheduler(init_gamma=0.95, max_gamma=0.995, horizon_ratio=1.75)

    print(f"🚀 Starting DUELING + NOISY + DOUBLE DQN training with {NUM_ENVS} envs...")
    start_time = time.time()

    # Initial Observation
    raw_boards = vec_env.reset()
    hint_sets = vec_env.get_hint_sets()
    states = prepare_state_batch(raw_boards, hint_sets)
    episode_count = start_episode

    while True:
        total_steps += NUM_ENVS
        
        # 1. SELECT ACTION (Argmax only - Exploration handled by NoisyNet)
        valid_masks = vec_env.valid_moves_batch()
        valid_masks_t = torch.tensor(valid_masks, device=device, dtype=torch.bool)
        
        # Reset Noise for new action selection
        policy_net.train() 
        policy_net.reset_noise()
        
        with torch.no_grad():
            states_t = torch.tensor(states, dtype=torch.float32, device=device)
            q_values_all = policy_net(states_t)
            q_values_all[~valid_masks_t] = -float('inf') 
            
            actions = q_values_all.argmax(dim=1).cpu().numpy().tolist()

        # 2. STEP 
        next_boards, rewards, dones = vec_env.step(actions)

        # --- REWARD SCALING ---
        # Thu nhỏ reward để ổn định Loss (tránh nổ gradient)
        rewards = np.array(rewards) * 0.1  

        next_hint_sets = vec_env.get_hint_sets()
        next_states = prepare_state_batch(next_boards, next_hint_sets)
        
        # --- ADAPTIVE GAMMA UPDATE ---
        # Lấy danh sách moves của các game vừa kết thúc để tính toán lại Gamma
        recent_finished_moves = []
        
        # 4. LOGGING & COLLECT METRICS
        for i in range(NUM_ENVS):
            env_curr['moves'][i] += 1
            env_curr['rewards'][i] += rewards[i]
            env_curr['max_tile'][i] = max(env_curr['max_tile'][i], max(next_boards[i]))
            
            if dones[i]:
                # Thu thập moves để update Gamma
                recent_finished_moves.append(env_curr['moves'][i])
                
                episode_count += 1
                metrics['r'].append(env_curr['rewards'][i])
                metrics['m'].append(env_curr['moves'][i])
                metrics['t'].append(env_curr['max_tile'][i])
                
                # Reset stats
                env_curr['moves'][i] = 0
                env_curr['rewards'][i] = 0
                env_curr['max_tile'][i] = 0
        
        # Cập nhật Gamma mới dựa trên phong độ vừa rồi
        if recent_finished_moves:
            curr_gamma = gamma_scheduler.step(recent_finished_moves)
        else:
            curr_gamma = gamma_scheduler.get_gamma()

        # CẬP NHẬT GAMMA CHO N-STEP BUFFER (Quan trọng!)
        # Để buffer tính đúng Discounted Return cho các bước đang chờ
        n_step_buffer.gamma = curr_gamma 

        # 3. N-STEP BUFFERING
        valid_transitions = n_step_buffer.add(states, actions, rewards, next_states, dones)
        
        if valid_transitions:
            batch_s, batch_a, batch_r, batch_ns, batch_d = zip(*valid_transitions)
            memory.push_batch(batch_s, batch_a, batch_r, batch_ns, batch_d)
        
        states = next_states

        # LOGGING PRINT
        if episode_count % 100 == 0 and recent_finished_moves: # Chỉ in khi có done mới tránh spam
            elapsed = time.time() - start_time
            fps = total_steps / elapsed
            print(f"Ep {episode_count:6d} | Steps: {total_steps:9d} | "
                  f"Avg R: {np.mean(metrics['r']):6.2f} | "
                  f"Moves: {np.mean(metrics['m']):4.1f} | "
                  f"MaxTile: {np.mean(metrics['t']):4.0f} | "
                  f"Loss: {np.mean(metrics['l']) if metrics['l'] else 0:.4f} | "
                  f"Q: {np.mean(metrics['q']) if metrics['q'] else 0:.2f} | "
                  f"Gamma: {curr_gamma:.4f} | " 
                  f"FPS: {fps:.1f}")
        
        if episode_count % 5000 == 0 and recent_finished_moves:
            save_checkpoint(episode_count, total_steps, policy_net, optimizer)

        # 5. TRAIN
        if len(memory.buffer) >= BATCH_SIZE:
            for _ in range(8):
                transitions = memory.sample(BATCH_SIZE)
                b_s, b_a, b_r, b_ns, b_d = transitions
                
                # Augmentation
                b_s, b_a, b_ns = augmenter.augment_tensors(b_s, b_a, b_ns)

                # Policy Net
                policy_net.train() 
                policy_net.reset_noise()
                q_eval = policy_net(b_s).gather(1, b_a.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    # Double DQN Logic
                    policy_net.eval() 
                    next_actions = policy_net(b_ns).argmax(1).unsqueeze(1)
                    policy_net.train() 
                    
                    # Target Evaluation
                    next_q = target_net(b_ns).gather(1, next_actions).squeeze(1)
                    next_q[b_d.bool()] = 0.0
                    
                    # --- DÙNG DYNAMIC GAMMA Ở ĐÂY ---
                    # Công thức: R + (gamma^N) * Q_max
                    discount = curr_gamma ** N_STEPS
                    expected_q = b_r + (discount * next_q)
                
                loss = criterion(q_eval, expected_q)
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                
                optimizer.step()
                
                metrics['l'].append(loss.item())
                metrics['q'].append(q_eval.mean().item())

        if total_steps % TARGET_UPDATE < NUM_ENVS:
            target_net.load_state_dict(policy_net.state_dict())
