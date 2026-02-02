import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gymnasium as gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# --- IMPORT HOẶC ĐỊNH NGHĨA LẠI KIẾN TRÚC MẠNG CỦA BẠN ---
# (Để script này chạy độc lập, tôi paste lại class của bạn vào đây)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F

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
            nn.Linear(14, 64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU()
        )
        self.hint_out_dim = 64
        combined_dim = self.board_out_dim + self.hint_out_dim
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.GELU()
        )

    def forward(self, observations):
        board = observations["board"].float()
        hint = observations["hint"].float() # Ensure float here too
        board_feat = self.resnet(board).flatten(1)
        hint_feat = self.hint_net(hint)
        combined = torch.cat((board_feat, hint_feat), dim=1)
        return self.fusion(combined)

# --- PHẦN 1: DATASET CLASS ---
class ExpertDataset(Dataset):
    def __init__(self, episodes):
        # episodes là list các tuple (obs_board, obs_hint, action)
        self.data = []
        print("Đang chuẩn bị dữ liệu cho PyTorch...")
        for ep in episodes:
            for step in ep['steps']:
                self.data.append(step)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        board, hint, action = self.data[idx]
        return {
            'board': torch.tensor(board, dtype=torch.float32),
            'hint': torch.tensor(hint, dtype=torch.float32),
            'action': torch.tensor(action, dtype=torch.long)
        }

# --- PHẦN 2: GENERATOR (THU THẬP DỮ LIỆU) ---
def generate_data(model, env, n_games=1000):
    """
    Chạy n_games song song và trả về list các episodes.
    Mỗi episode chứa: {'max_tile': int, 'length': int, 'steps': [(board, hint, action), ...]}
    """
    n_envs = env.num_envs
    # Buffer tạm cho từng env
    current_episodes = [ {'steps': [], 'max_tile': 0} for _ in range(n_envs) ]
    finished_episodes = []
    
    obs = env.reset()

    # --- KHỞI TẠO MASK LẦN ĐẦU ---
    current_masks = np.array(env.env_method("valid_action_mask"))
    
    pbar = tqdm(total=n_games, desc="Generating Games")
    
    while len(finished_episodes) < n_games:
        # Truyền mask đã có vào predict
        actions, _ = model.predict(obs, action_masks=current_masks, deterministic=False)
        
        # Step môi trường
        next_obs, rewards, dones, infos = env.step(actions)

        next_masks = []
        
        for i in range(n_envs):
            # 1. Lưu dữ liệu training
            board_data = obs['board'][i].copy()
            hint_data = obs['hint'][i].copy()
            act = actions[i]
            
            # Lưu step
            current_episodes[i]['steps'].append((board_data, hint_data, act))
            
            # Update Score
            # Giả sử info['score'] là reward tích lũy, hãy chắc chắn wrapper gửi đúng
            current_episodes[i]['total_reward'] = infos[i]['score']
            
            # 2. Xử lý Mask (QUAN TRỌNG)
            if dones[i]:
                # Game kết thúc -> Lưu vào kho
                finished_episodes.append({
                    'steps': current_episodes[i]['steps'],
                    'score': current_episodes[i]['total_reward'], 
                    'real_game_score': infos[i].get('real_score', 0)
                })
                
                # Reset buffer
                current_episodes[i] = {'steps': [], 'total_reward': 0.0}
                pbar.update(1)
                
                # --- FIX CRASH HERE ---
                # Vì env đã auto-reset, info['action_mask'] là của ván cũ (đã chết).
                # Ta cần mask cho ván mới (next_obs[i]).
                # Gọi env_method chỉ cho index i để lấy mask chuẩn.
                new_mask = env.env_method("valid_action_mask", indices=i)[0]
                next_masks.append(new_mask)
            else:
                # Nếu chưa chết, mask trong info là đúng cho bước tiếp theo
                next_masks.append(infos[i]['action_mask'])
        
        # Cập nhật state cho vòng sau
        obs = next_obs
        current_masks = np.array(next_masks)
        
        if len(finished_episodes) >= n_games:
            break
            
    pbar.close()
    return finished_episodes

# --- PHẦN 3: MAIN FLOW ---
def main():
    # 1. Cấu hình
    MODEL_PATH = "logs_ppo_threes_resnet/ppo_resnet_14200000_steps.zip" # Đường dẫn model cũ
    NEW_MODEL_PATH = "logs_ppo_threes_resnet/ppo_resnet_EVOLVED.zip"
    
    N_GAMES = 100000      # Số ván chơi thử (Nên để 10k - 100k)
    TOP_PERCENT = 0.05   # Lấy top 5% (Elite)
    EPOCHS = 3           # Số vòng train Supervised
    BATCH_SIZE = 256
    LR = 3e-4
    
    # 2. Load Model & Env
    try:
        from threes_maskable_ppo_train import make_env 
    except ImportError:
        print("Copy file này vào cùng thư mục với threes_maskable_ppo_train.py")
        return

    print(f"Loading Teacher Model: {MODEL_PATH}")
    model = MaskablePPO.load(MODEL_PATH)
    
    # Tạo môi trường song song để chạy cho nhanh
    # Lưu ý: make_env phải trả về function tạo env
    env = SubprocVecEnv([make_env for _ in range(8)]) # 8 luồng
    
    # 3. Generate Data (Teacher Plays)
    print(f"Phase 1: Generating {N_GAMES} games...")
    all_episodes = generate_data(model, env, N_GAMES)
    
    # 4. Filter Elite (Lọc Vàng)
    print("Phase 2: Filtering Elite Games...")
    # Sắp xếp: Ưu tiên Score cao (hoặc Length dài)
    all_episodes.sort(key=lambda x: x['score'], reverse=True)
    
    n_elite = int(N_GAMES * TOP_PERCENT)
    elite_episodes = all_episodes[:n_elite]
    
    print(f"--> Đã chọn {n_elite} ván xuất sắc nhất.")
    print(f"--> Best Score: {elite_episodes[0]['score']}")
    print(f"--> Worst Elite Score: {elite_episodes[-1]['score']}")
    
    # Giải phóng RAM (Xóa data rác)
    del all_episodes
    
    # 5. Train Student (Supervised Learning)
    print("Phase 3: Training Student (Cloning)...")
    
    # Chuyển data thành PyTorch Dataset
    dataset = ExpertDataset(elite_episodes)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Trích xuất mạng Policy từ Model PPO ra để train
    # model.policy là một nn.Module, ta có thể train nó trực tiếp!
    policy_network = model.policy
    policy_network.train()
    
    optimizer = optim.Adam(policy_network.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
            boards = batch['board'].to(model.device)
            hints = batch['hint'].to(model.device)
            target_actions = batch['action'].to(model.device)
            
            # Forward pass
            # SB3 policy forward trả về (features, values, log_prob)
            # Chúng ta cần logits hoặc distribution.
            # Cách dễ nhất trong SB3: get_distribution
            
            # policy.extract_features trả về features
            features = policy_network.extract_features({'board': boards, 'hint': hints})
            
            # policy.action_net là lớp Linear cuối cùng ra Logits
            # (Lưu ý: trong MaskablePPO cấu trúc có thể hơi khác, ta dùng get_distribution cho chuẩn)
            
            # Tuy nhiên, để train Supervised, ta cần Logits thô (chưa qua Softmax/Masking)
            # Mạng MaskablePPO tách Action Masking ra ngoài mạng neural.
            # Mạng neural chỉ output logits gốc.
            latent_pi, _ = policy_network.mlp_extractor(features)
            logits = policy_network.action_net(latent_pi)
            
            # Tính Loss: So sánh Logits dự đoán với Action thực tế của Teacher
            loss = criterion(logits, target_actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} Avg Loss: {total_loss / len(dataloader):.4f}")

    # 6. Save Result
    print(f"Phase 4: Saving Evolved Model to {NEW_MODEL_PATH}")
    model.save(NEW_MODEL_PATH)
    print("Done! Bạn có thể dùng model này làm Teacher cho vòng lặp tiếp theo.")
    
    env.close()

if __name__ == "__main__":
    main()