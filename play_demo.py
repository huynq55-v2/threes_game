import torch
import numpy as np
import os
import time
import threes_rs # Rust binding

# Import from training script
try:
    from train_kaggle import DuelingNoisyDQN, TILE_MAP, prepare_state_batch
except ImportError:
    print("❌ Cannot import 'train_kaggle.py'. Make sure it is in the same directory.")
    exit(1)



def main():
    # 1. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DuelingNoisyDQN().to(device)
    model.eval() # Important: Disable Noise for consistent play
    
    # Pytorch's load_state_dict handles strict=True by default, ensuring structure matches
    
    # Search for checkpont
    # Search for checkpont
    model_path = "threes_residual_net.pth"
    candidates = [
        "threes_residual_net.pth",
        "threes_dueling_noisy_double_dqn_latest.pth",
        "/kaggle/working/threes_residual_net.pth",
        "/kaggle/input/threes-binary/threes_residual_net.pth"
    ]
    
    found = False
    for c in candidates:
        if os.path.exists(c):
            model_path = c
            found = True
            break
    
    if found:
        print(f"Loading model from {model_path}...")
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state'])
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print(f"Warning: Model file not found! Playing with random initialization.")
        time.sleep(2)

    # 2. Init Game
    try:
        env = threes_rs.ThreesEnv()
    except Exception as e:
        print(f"Failed to init Env: {e}")
        return
    
    # 3. Init UI
    try:
        env.init_ui()
    except Exception as e:
        print(f"Failed to init UI: {e}")
        return

    error = None 
    try:
        board = env.reset()
        done = False
        action_names = ["Up", "Down", "Left", "Right"]
        
        env.render_cli("Manual Mode: Arrows/WASD to Move. Scroll Up/Down to Undo/Redo. 'Q' to quit.")
        
        while not done:
            # Wait for user input
            # get_input returns Option<u32> (None = Quit, 10=Undo, 11=Redo)
            try:
                action = env.get_input()
            except Exception as e:
                # Should not happen usually
                error = e
                break
            
            if action is None:
                break
            
            if action == 10: # Undo
                board = env.undo()
                env.render_cli("Undid last move.")
                continue
            elif action == 11: # Redo
                board = env.redo()
                env.render_cli("Redid move.")
                continue
                
            # Step
            # Note: Rust Env expects simple integer action
            board, reward, done = env.step(action)
            
            # Format UI Message
            if action < 4:
                status_msg = f"Last Action: {action_names[action]} | Reward: {reward:.2f}"
            else:
                status_msg = f"Action: {action}"
            
            # Render via Rust
            env.render_cli(status_msg)
            
            if done:
                env.render_cli("GAME OVER! Press Ctrl+C to exit.")
                time.sleep(5)
                
    except KeyboardInterrupt:
        pass
    except Exception as e:
        error = e
    finally:
        env.cleanup_ui()
        if error:
            print(f"An error occurred: {error}")
        print("Game closed.")

if __name__ == "__main__":
    main()
