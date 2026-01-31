use threes_rs::game::Game;
use threes_rs::ui::{ConsoleUI, InputEvent};

fn main() -> std::io::Result<()> {
    // 1. Initialize Terminal (Raw Mode, Alt Screen)
    ConsoleUI::init()?;

    // Ensure cleanup on exit
    // We can't catch panics easily without a hook, but for normal flow:
    let result = run_game();

    // 2. Cleanup
    ConsoleUI::cleanup()?;

    result
}

fn run_game() -> std::io::Result<()> {
    let mut game = Game::new();
    let mut message = String::new();

    loop {
        // 3. Calc Valid Moves
        let valid_moves = game.get_valid_moves();

        // 4. Draw
        let msg_display = if message.is_empty() {
            None
        } else {
            Some(message.as_str())
        };
        ConsoleUI::print_game_state(&game, &valid_moves, msg_display)?;

        // Reset message after one frame
        message.clear();

        // 5. Check Game Over
        if game.check_game_over() {
            // Show Final State with Game Over Message
            ConsoleUI::print_game_state(
                &game,
                &[],
                Some("GAME OVER! Press ENTER to Restart, Q to Quit."),
            )?;

            // Wait for Enter or Q
            // Simple wait helper needed? Or just reuse get_input but specific?
            // Let's implement a simple wait loop or just use existing.
            // Actually, we want to wait specifically for Enter or Q.
            ConsoleUI::wait_for_enter()?;

            // Restart
            game = Game::new();
            continue;
        }

        // 6. Input
        // This blocks until a directional key or Quit is pressed
        match ConsoleUI::get_input()? {
            InputEvent::Dir(dir) => {
                if valid_moves.contains(&dir) {
                    let moved = game.move_dir(dir);
                    if !moved {
                        message = "Move failed (unexpected).".to_string();
                    }
                } else {
                    message = "Invalid Move! Cannot move that way.".to_string();
                }
            }
            InputEvent::Undo | InputEvent::Redo => {
                message = "History not supported in CLI mode yet.".to_string();
            }
            InputEvent::Quit => {
                break;
            }
        }
    }
    Ok(())
}
