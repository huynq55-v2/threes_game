from stable_baselines3.common.env_checker import check_env
import numpy as np
# Import class env của bạn
# from your_file import ThreesGymEnv, make_env 
from threes_gym import ThreesGymEnv

print("🔍 Đang chạy check_env của SB3...")
try:
    # Khởi tạo env
    env = ThreesGymEnv()
    
    # Hàm này sẽ crash ngay nếu môi trường không đúng chuẩn Gym
    check_env(env, warn=True)
    
    print("✅ Environment Check: PASSED! Cấu trúc môi trường có vẻ ổn.")
except Exception as e:
    print(f"❌ Environment Check: FAILED! Lỗi: {e}")