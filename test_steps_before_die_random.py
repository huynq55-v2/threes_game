import numpy as np
import matplotlib.pyplot as plt
import threes_rs

def run_statistical_test(env, num_episodes=1000000):
    stats_tile_6 = []

    print(f"Bắt đầu chạy {num_episodes} ván test (Random Valid Moves)...")

    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        while not done:
            masks = env.valid_moves() 
            valid_indices = [idx for idx, can_move in enumerate(masks) if can_move]
            
            if not valid_indices:
                break
                
            action = np.random.choice(valid_indices)
            board_flat, _, terminated, _ = env.step(action)
            done = terminated
            steps += 1
        
        max_tile = max(board_flat)
        if max_tile == 6:
            stats_tile_6.append(steps)

        if i % 10000 == 0 and i > 0:
            print(f"Đã hoàn thành {i} ván...")

    if not stats_tile_6:
        print("\nKhông có ván nào kết thúc ở Max Tile 6.")
        return None

    # --- LOGIC PHÂN TÍCH MỚI ---
    data = np.array(stats_tile_6)
    p95 = np.percentile(data, 95)
    p99 = np.percentile(data, 99)
    
    print("\n" + "="*30)
    print("--- KẾT QUẢ PHÂN TÍCH THỰC TẾ ---")
    print(f"Số ván đạt Max 6: {len(data)} ({len(data)/num_episodes*100:.2f}%)")
    print(f"Min steps: {np.min(data)}")
    print(f"Max steps: {np.max(data)}")
    print(f"Mean steps: {np.mean(data):.2f}")
    print("-" * 30)
    print(f"Ngưỡng 95% (Top 5% tệ nhất): {p95:.2f} bước")
    print(f"Ngưỡng 99% (Chắc chắn lỗi):   {p99:.2f} bước")
    print("="*30)

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=range(min(data), max(data) + 2), alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(p95, color='orange', linestyle='--', label=f'95th Percentile ({p95:.1f})')
    plt.axvline(p99, color='red', linestyle='-', label=f'99th Percentile ({p99:.1f})')
    
    plt.title("Phân phối số bước đi khi kết thúc ván ở Max Tile 6")
    plt.xlabel("Số bước (Steps)")
    plt.ylabel("Số lượng ván đấu")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    return data

if __name__ == "__main__":
    env = threes_rs.ThreesEnv() 
    stats = run_statistical_test(env)