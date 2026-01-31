import os
import sys
import copy
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

# --- PHẦN 1: INPUT KEYBOARD ---
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

get_key = _Getch()

# --- PHẦN 2: TIME MACHINE ---
class TimeMachine:
    def __init__(self):
        self.timeline = []
        self.cursor = -1

    def save(self, env, obs):
        # Rust module handles history automatically in step()
        pass

    def undo(self, env):
        try:
            # env.unwrapped.game is the Rust ThreesEnv object
            # It returns (board_flat, reward, done, hint_list)
            res = env.unwrapped.game.undo()
            board, _, _, hint = res
            
            # Reconstruct observation using helper in ThreesGymEnv
            obs = env.unwrapped._process_obs(board, hint)
            return env, obs
        except Exception as e:
            # e.g. history empty or panic
            return None, None

    def redo(self, env):
        try:
            res = env.unwrapped.game.redo()
            board, _, _, hint = res
            obs = env.unwrapped._process_obs(board, hint)
            return env, obs
        except Exception as e:
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

# --- PHẦN 4: UI MỚI (SOLID BLOCK) ---

BG_STYLES = {
    0:  Back.BLACK,
    1:  Back.CYAN,
    2:  Back.RED,
    3:  Back.WHITE,
    6:  Back.WHITE,
    12: Back.WHITE,
    24: Back.WHITE,
    48: Back.YELLOW,             
    96: Back.YELLOW,
    192: Back.YELLOW,
    384: Back.BLUE,
    768: Back.MAGENTA,
    1536: Back.GREEN,
    3072: Back.GREEN,
    6144: Back.BLACK
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

# --- UI DRAWING (Đã sửa để nhận đúng tham số) ---
def draw_ui(env, obs, message=""):
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Lấy board và hint trực tiếp từ env và obs
    board = np.array(env.unwrapped.game.get_board_flat()).reshape(4,4)
    hint_idx = np.argmax(obs['hint'])
    TILE_MAP_INV = {i: v for v, i in env.unwrapped.TILE_MAP.items()}
    hint_val = TILE_MAP_INV.get(hint_idx, 0)
    
    print(f"{Style.BRIGHT}=== THREES! AI GYM (IRON DISCIPLINE) ==={Style.RESET_ALL}\n")

    tile_width = 6 
    for row in board:
        # Line 1, 2, 3: Vẽ khối màu solid
        for line_type in range(3):
            for val in row:
                bg, fg = get_style(val) # Hàm get_style bạn đã định nghĩa
                if line_type == 1: # Dòng giữa có số
                    s_val = "." if val == 0 else str(val)
                    print(f"{bg}{fg}{s_val:^{tile_width}}{Style.RESET_ALL}", end=" ")
                else: # Dòng đệm trên/dưới
                    print(f"{bg}{' ' * tile_width}{Style.RESET_ALL}", end=" ")
            print()
        print() 

    # Vẽ Hint
    bg_h, fg_h = get_style(hint_val)
    hint_str = str(hint_val) if hint_val > 0 else "?"
    print(f"Next Tile:")
    print(f"{bg_h}{' ' * tile_width}{Style.RESET_ALL}")
    print(f"{bg_h}{fg_h}{hint_str:^{tile_width}}{Style.RESET_ALL}")
    print(f"{bg_h}{' ' * tile_width}{Style.RESET_ALL}")
    
    print(f"\n{Fore.YELLOW}Arrows{Style.RESET_ALL}: Move | {Fore.GREEN}U{Style.RESET_ALL}: Undo | {Fore.RED}Q{Style.RESET_ALL}: Quit")
    if message: print(f"\n{message}")
    sys.stdout.flush()

# --- PHẦN 5: MAIN ---
def main():
    model_path = "./logs_ppo_threes_resnet/ppo_resnet_6400000_steps.zip"
    try:
        model = MaskablePPO.load(model_path)
    except:
        print("Lỗi load model!")
        return

    env = make_env()
    obs, _ = env.reset()
    tm = TimeMachine() # Sử dụng TimeMachine đã fix ở các version trước
    tm.save(env, obs)
    
    msg = ""

    while True:
        # GỌI ĐÚNG THAM SỐ: Chỉ truyền env, obs và msg
        draw_ui(env, obs, msg)
        msg = ""

        key = get_key()
        if key in ['q', 'Q']: break
        
        # Xử lý Undo
        if key in ['u', 'U']:
            e, o = tm.undo(env)
            if e:
                obs = o
                msg = f"{Fore.GREEN}Đã Undo."
            continue

        # Xử lý Move
        action_map = {'UP': 0, 'DOWN': 1, 'LEFT': 2, 'RIGHT': 3}
        if key in action_map:
            action = action_map[key]
            masks = env.unwrapped.valid_action_mask()
            
            if not masks[action]:
                msg = f"{Fore.RED}Bị tường chặn!"
                continue

            # --- LOGIC KIỂM DUYỆT CỦA AI ---
            # Chỉ kích hoạt khi đã có số >= 48
            current_max = max(env.unwrapped.game.get_board_flat())
            if current_max >= 12:
                is_weak, score = get_ai_evaluation(model, obs, masks, action)
                if is_weak:
                    # KHÔNG CHO ĐI: Chỉ hiện thông báo và bắt chọn lại
                    msg = f"{Back.RED}{Fore.WHITE} WEAK MOVE ({key})! ({score:.2f}) {Style.RESET_ALL} {Fore.YELLOW}AI chặn. Thử hướng khác!{Style.RESET_ALL}"
                    continue

            # Thực thi nếu không yếu
            obs, _, done, _, _ = env.step(action)
            tm.save(env, obs)
            
            if done:
                draw_ui(env, obs, f"{Back.RED} GAME OVER! {Style.RESET_ALL}")
                print("Bấm U để Undo hoặc Q để Quit.")
                while True:
                    k = get_key().lower()
                    if k == 'u': 
                        e, o = tm.undo(env)
                        if e: obs = o; break
                    if k == 'q': return

if __name__ == "__main__":
    main()