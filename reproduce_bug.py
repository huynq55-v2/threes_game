import torch
import numpy as np
from threes_maskable_ppo_train import make_env
from sb3_contrib import MaskablePPO
import torch.nn.functional as F

def reproduce():
    # 1. Khởi tạo
    model = MaskablePPO.load("./logs_ppo_threes_resnet/ppo_resnet_6400000_steps.zip")
    env = make_env()
    obs, _ = env.reset()

    # 2. Tái hiện bàn cờ bạn đã cung cấp
    # 0 0 2 0
    # 0 2 0 3
    # 0 0 1 1
    # 3 6 3 1
    target_board = [
        0, 0, 2, 0,
        0, 2, 0, 3,
        0, 0, 1, 1,
        3, 6, 3, 1
    ]
    
    # Ép môi trường sang trạng thái này
    # Giả sử next_value là 1 (con số tiếp theo sẽ xuất hiện)
    env.unwrapped.game.set_board(target_board, 10, 1)
    
    # Lấy observation mới sau khi set board
    # Lưu ý: Bạn cần gọi hàm xử lý obs giống hệt lúc train
    # Ở đây tôi giả định bạn có hàm _process_obs hoặc tương đương
    current_board = env.unwrapped.game.get_board_flat()
    current_hint = env.unwrapped.game.get_hint_set()
    obs = env.unwrapped._process_obs(current_board, current_hint)

    # 3. Chẩn đoán "Trắng đen"
    with torch.no_grad():
        # Chuyển đổi observation sang Tensor và đưa vào thiết bị (CPU/GPU)
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        
        # A. Lấy Mask từ Rust
        masks = env.unwrapped.valid_action_mask()
        # Chuyển mask sang tensor để SB3 xử lý nếu cần
        mask_tensor = torch.as_tensor(masks).bool().view(1, -1)
        
        # B. Lấy Raw Logits (ĐÃ SỬA LỖI Ở ĐÂY)
        # 1. Trích xuất đặc trưng từ Observation
        features = model.policy.extract_features(obs_tensor)
        # 2. Đưa qua mạng Latent (chỉ lấy phần của Actor - Pi)
        latent_pi, _ = model.policy.mlp_extractor(features)
        # 3. Đưa qua lớp Linear cuối cùng để ra Logits
        logits = model.policy.action_net(latent_pi)
        
        # Tính xác suất thô bằng Softmax (chưa bị Mask can thiệp)
        raw_probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # C. Lấy Final Probs (Sau khi áp Mask)
        # SB3 MaskablePPO áp dụng mask ngay trong hàm get_distribution
        dist = model.policy.get_distribution(obs_tensor, action_masks=mask_tensor)
        final_probs = dist.distribution.probs.cpu().numpy()[0]

    # 4. In kết quả
    actions = ["UP", "DOWN", "LEFT", "RIGHT"]
    print(f"\n{'Action':<8} | {'Mask':<5} | {'Raw Prob':<10} | {'Final Prob'}")
    print("-" * 50)
    for i in range(4):
        print(f"{actions[i]:<8} | {str(masks[i]):<5} | {raw_probs[i]:.6f} | {final_probs[i]:.6f}")

if __name__ == "__main__":
    reproduce()