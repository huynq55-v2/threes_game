import os
import sys
import copy
import time
import numpy as np
import torch
import colorama
from colorama import Fore, Back, Style

# Import môi trường
try:
    from threes_maskable_ppo_train import make_env, ThreesGymEnv
except ImportError:
    print("Lỗi: Không tìm thấy file 'threes_maskable_ppo_train.py'.")
    exit()

from sb3_contrib import MaskablePPO

colorama.init(autoreset=True)

# --- PHẦN 1: INPUT KEYBOARD & CHECKING ---
class _Getch:
    def __init__(self):
        try: self.impl = _GetchWindows()
        except ImportError: self.impl = _GetchUnix()
    def __call__(self): return self.impl()

class _GetchUnix:
    def __init__(self): import tty, sys
    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                ch2 = sys.stdin.read(1)
                if ch2 == '[':
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'A': return 'UP'
                    if ch3 == 'B': return 'DOWN'
                    if ch3 == 'C': return 'RIGHT'
                    if ch3 == 'D': return 'LEFT'
            return ch
        finally: termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

class _GetchWindows:
    def __init__(self): import msvcrt
    def __call__(self):
        import msvcrt
        ch = msvcrt.getch()
        if ch == b'\xe0' or ch == b'\x00':
            ch = msvcrt.getch()
            if ch == b'H': return 'UP'
            if ch == b'P': return 'DOWN'
            if ch == b'K': return 'LEFT'
            if ch == b'M': return 'RIGHT'
        return ch.decode('utf-8')

def kbhit():
    if os.name == 'nt':
        import msvcrt
        return msvcrt.kbhit()
    else:
        import select
        dr, dw, de = select.select([sys.stdin], [], [], 0)
        return dr != []

get_key = _Getch()

# --- PHẦN 2: TIME MACHINE ---
class TimeMachine:
    def __init__(self):
        self.timeline = []

    def save(self, env, obs):
        pass

    def undo(self, env):
        try:
            res = env.unwrapped.game.undo()
            board, _, _, hint = res
            obs = env.unwrapped._process_obs(board, hint)
            return env, obs
        except Exception:
            return None, None

    def redo(self, env):
        try:
            res = env.unwrapped.game.redo()
            board, _, _, hint = res
            obs = env.unwrapped._process_obs(board, hint)
            return env, obs
        except Exception:
            return None, None

# --- PHẦN 3: AI CHECK ---
def get_ai_evaluation(model, obs, action_masks, user_action):
    with torch.no_grad():
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.cpu().numpy()[0]
        masked_probs = probs * action_masks
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
        user_prob = masked_probs[user_action]
        lower_mass = np.sum(masked_probs[masked_probs <= user_prob])
        return lower_mass < 0.5, lower_mass

# --- PHẦN 4: UI MỚI (FIXED NEXT TILE) ---

BG_STYLES = {
    0:  Back.BLACK, 1:  Back.CYAN, 2:  Back.RED, 3:  Back.WHITE,
    6:  Back.WHITE, 12: Back.WHITE, 24: Back.WHITE, 48: Back.YELLOW,             
    96: Back.YELLOW, 192: Back.YELLOW, 384: Back.BLUE, 768: Back.MAGENTA,
    1536: Back.GREEN, 3072: Back.GREEN, 6144: Back.BLACK
}

FG_STYLES = {
    0:  Fore.BLACK, 1:  Fore.WHITE, 2:  Fore.WHITE, 3:  Fore.BLACK,
    384: Fore.WHITE, 768: Fore.WHITE, 1536: Fore.BLACK, 6144: Fore.WHITE
}

def get_style(val):
    bg = BG_STYLES.get(val, Back.MAGENTA)
    if val >= 3 and val < 384: fg = Fore.BLACK
    else: fg = FG_STYLES.get(val, Fore.WHITE)
    return bg, fg

def draw_ui(env, obs, message="", is_autoplay=False):
    os.system('cls' if os.name == 'nt' else 'clear')
    
    board = np.array(env.unwrapped.game.get_board_flat()).reshape(4,4)
    
    # --- BUG FIX SỐ 4: HIỂN THỊ HẾT CÁC HINT ---
    # Lấy map từ index ngược ra value
    TILE_MAP_INV = {i: v for v, i in env.unwrapped.TILE_MAP.items()}
    
    # Tìm tất cả các index có giá trị > 0 trong vector hint
    hint_indices = np.where(obs['hint'] > 0.5)[0]
    
    # Convert sang list các giá trị thật
    hint_values = [TILE_MAP_INV.get(idx, 0) for idx in hint_indices]
    hint_values.sort() # Sắp xếp cho đẹp
    
    # Tạo chuỗi hiển thị (ví dụ: "6 | 12")
    if not hint_values:
        hint_str = "?"
        bg_h, fg_h = Back.BLACK, Fore.WHITE
    else:
        hint_str = " | ".join(str(v) for v in hint_values)
        # Lấy màu của con đầu tiên để làm nền đại diện
        bg_h, fg_h = get_style(hint_values[0])
    # -------------------------------------------

    print(f"{Style.BRIGHT}=== THREES! AI GYM (IRON DISCIPLINE) ==={Style.RESET_ALL}\n")

    tile_width = 8 # Tăng width lên chút để hiển thị số to
    for row in board:
        for line_type in range(3):
            for val in row:
                bg, fg = get_style(val)
                if line_type == 1:
                    s_val = "." if val == 0 else str(val)
                    print(f"{bg}{fg}{s_val:^{tile_width}}{Style.RESET_ALL}", end=" ")
                else:
                    print(f"{bg}{' ' * tile_width}{Style.RESET_ALL}", end=" ")
            print()
        print() 

    # Vẽ Hint Box rộng hơn chút để chứa nhiều số
    hint_width = max(tile_width, len(hint_str) + 2)
    print(f"Next Tile:")
    print(f"{bg_h}{' ' * hint_width}{Style.RESET_ALL}")
    print(f"{bg_h}{fg_h}{hint_str:^{hint_width}}{Style.RESET_ALL}")
    print(f"{bg_h}{' ' * hint_width}{Style.RESET_ALL}")
    
    status_ai = f"{Back.GREEN} ON {Style.RESET_ALL}" if is_autoplay else f"{Back.RED} OFF {Style.RESET_ALL}"

    print(f"\n{Fore.YELLOW}Arrows{Style.RESET_ALL}: Move | {Fore.GREEN}U{Style.RESET_ALL}: Undo | {Fore.RED}R{Style.RESET_ALL}: Restart")
    print(f"{Fore.CYAN}P{Style.RESET_ALL}: AI Autoplay [{status_ai}]")
    
    if message: print(f"\n{message}")
    sys.stdout.flush()

# --- PHẦN 5: MAIN ---
def main():
    model_path = "./logs_ppo_threes_resnet/ppo_resnet_14200000_steps.zip"
    try:
        model = MaskablePPO.load(model_path)
    except:
        print("Lỗi load model!")
        return

    env = make_env()
    obs, _ = env.reset()
    tm = TimeMachine()
    tm.save(env, obs)
    
    msg = ""
    ai_autoplay = False
    
    # Delay cho AI (giây) - Fix số 3
    AI_DELAY = 0.15 

    while True:
        draw_ui(env, obs, msg, is_autoplay=ai_autoplay)
        msg = ""

        # --- MODE 1: AI AUTOPLAY ---
        if ai_autoplay:
            # Check phím chen ngang (Non-blocking)
            if kbhit():
                key = get_key()
                if key in ['p', 'P']:
                    ai_autoplay = False
                    msg = f"{Fore.YELLOW}Đã tắt AI Autoplay."
                    continue 
                if key in ['q', 'Q']: return
                if key in ['r', 'R']: # Restart ngay cả khi AI đang chạy
                     obs, _ = env.reset()
                     tm = TimeMachine()
                     tm.save(env, obs)
                     msg = f"{Fore.GREEN}Đã Restart Game Mới!"
                     continue

            # AI Logic
            masks = env.unwrapped.valid_action_mask()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            
            obs, _, done, _, _ = env.step(action)
            tm.save(env, obs)
            
            if done:
                ai_autoplay = False # Tự tắt AI khi chết
                draw_ui(env, obs, f"{Back.RED} GAME OVER! {Style.RESET_ALL}", is_autoplay=False)
                print("\nBấm R để Restart, U để Undo, Q để Quit.")
                
                # Loop chờ xử lý sau khi chết
                while True:
                    k = get_key().lower()
                    if k == 'q': return
                    if k == 'u': 
                        e, o = tm.undo(env)
                        if e: obs = o; break
                    if k == 'r': # Fix số 2: Restart sau Game Over
                        obs, _ = env.reset()
                        tm = TimeMachine()
                        tm.save(env, obs)
                        break
                continue # Quay lại vòng lặp chính (với board mới hoặc undo)

            time.sleep(AI_DELAY) # Fix số 3: Delay
            continue # BUG FIX SỐ 1: Continue ngay để AI chạy tiếp, không rơi xuống phần 'get_key' chặn ở dưới

        # --- MODE 2: MANUAL PLAYER (Blocking) ---
        
        key = get_key() # Chờ phím ở đây
        
        if key in ['q', 'Q']: break
        
        # BUG FIX SỐ 1: Bấm P là loop lại ngay để lọt vào block if ai_autoplay ở trên
        if key in ['p', 'P']: 
            ai_autoplay = True
            continue 

        # Fix số 2: Restart thủ công
        if key in ['r', 'R']:
            obs, _ = env.reset()
            tm = TimeMachine()
            tm.save(env, obs)
            msg = f"{Fore.GREEN}Đã Restart Game Mới!"
            continue

        if key in ['u', 'U']:
            e, o = tm.undo(env)
            if e:
                obs = o
                msg = f"{Fore.GREEN}Đã Undo."
            continue

        action_map = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3}
        if key in action_map:
            action = action_map[key]
            masks = env.unwrapped.valid_action_mask()
            
            if not masks[action]:
                msg = f"{Fore.RED}Bị tường chặn!"
                continue

            # AI Advisor check
            current_max = max(env.unwrapped.game.get_board_flat())
            if current_max >= 12:
                is_weak, score = get_ai_evaluation(model, obs, masks, action)
                if is_weak:
                    msg = f"{Back.RED}{Fore.WHITE} WEAK MOVE ({key})! ({score:.2f}) {Style.RESET_ALL} {Fore.YELLOW}AI chặn.{Style.RESET_ALL}"
                    continue

            obs, _, done, _, _ = env.step(action)
            tm.save(env, obs)
            
            if done:
                draw_ui(env, obs, f"{Back.RED} GAME OVER! {Style.RESET_ALL}", is_autoplay=False)
                print("\nBấm R để Restart, U để Undo, Q để Quit.")
                while True:
                    k = get_key().lower()
                    if k == 'q': return
                    if k == 'u': 
                        e, o = tm.undo(env)
                        if e: obs = o; break
                    if k == 'r': # Restart sau Game Over thủ công
                        obs, _ = env.reset()
                        tm = TimeMachine()
                        tm.save(env, obs)
                        break

if __name__ == "__main__":
    main()