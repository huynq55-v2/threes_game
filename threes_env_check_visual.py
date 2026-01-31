import time
from threes_gym import ThreesGymEnv
import numpy as np

def print_pretty_board(obs_board):
    """Hàm helper để in bàn cờ từ dạng Rank (0-15) về số thật (3, 6, 12...)"""
    # obs_board shape: (1, 4, 4) -> lấy (4, 4)
    ranks = obs_board[0]
    
    print("-" * 25)
    for row in ranks:
        line = []
        for r in row:
            if r == 0: val = 0
            elif r == 1: val = 1
            elif r == 2: val = 2
            else: val = int(3 * (2 ** (r - 3))) # Công thức ngược của log2
            line.append(f"{val:4}")
        print("|" + "|".join(line) + "|")
    print("-" * 25)

def test_game_loop():
    env = ThreesGymEnv()
    obs, info = env.reset()
    
    print("\n🎮 BẮT ĐẦU TEST GAME LOOP THỦ CÔNG")
    
    for i in range(10): # Test 10 bước
        print(f"\n--- STEP {i+1} ---")
        
        # 1. In bàn cờ hiện tại
        print("Board State:")
        print_pretty_board(obs['board'])
        
        # 2. Kiểm tra Hint
        hint_vec = obs['hint']
        hint_idx = np.argmax(hint_vec)
        # Mapping ngược lại từ index sang value (bạn cần check lại TILE_MAP của bạn)
        # Giả sử: 0->1, 1->2, 2->3...
        print(f"Hint Vector (One-hot argmax): {hint_idx}")
        
        # 3. Kiểm tra Action Mask (QUAN TRỌNG)
        mask = env.valid_action_mask()
        ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
        valid_actions_text = [ACTIONS[i] for i, v in enumerate(mask) if v]
        print(f"Valid Actions (Rust Mask): {mask} -> {valid_actions_text}")
        
        if not any(mask):
            print("☠️ GAME OVER (No valid moves)")
            break
            
        # 4. Chọn đại một action hợp lệ
        valid_indices = np.where(mask)[0]
        action = np.random.choice(valid_indices)
        print(f"👉 Selecting Action: {ACTIONS[action]}")
        
        # 5. Step
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"Reward nhận được: {reward}")
        
        if done:
            print("🏁 Game Done!")
            # In bàn cờ cuối cùng
            print_pretty_board(obs['board'])
            obs, _ = env.reset()
            print("🔄 Reset Game")

test_game_loop()