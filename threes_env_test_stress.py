from threes_gym import ThreesGymEnv
import time
import numpy as np

def stress_test():
    print("\n🏃 BẮT ĐẦU STRESS TEST (1000 steps)...")
    env = ThreesGymEnv()
    obs, _ = env.reset()
    start = time.time()
    
    total_reward = 0
    steps = 0
    
    try:
        for _ in range(1000):
            mask = env.valid_action_mask()
            # Chọn action ngẫu nhiên trong các action hợp lệ
            if not np.any(mask): # Game over nhưng chưa trả về done?
                action = 0 
            else:
                action = np.random.choice(np.where(mask)[0])
                
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                obs, _ = env.reset()
                
        end = time.time()
        print(f"✅ STRESS TEST PASSED!")
        print(f"Tốc độ: {steps / (end - start):.2f} steps/sec (Python + Rust overhead)")
        print(f"Tổng reward random: {total_reward}")
        
    except Exception as e:
        print(f"❌ CRASHED ở step {steps}!")
        print(f"Lỗi: {e}")

stress_test()