use macroquad::prelude::*;
use threes_rs::game::{Direction, Game};
use threes_rs::n_tuple_network::NTupleNetwork;
use threes_rs::threes_env::ThreesEnv;

// ============================================================================
// CONSTANTS
// ============================================================================
const TILE_SIZE: f32 = 100.0;
const PADDING: f32 = 12.0;
const BOARD_OFFSET_X: f32 = 40.0;
const BOARD_OFFSET_Y: f32 = 140.0;
const EPSILON: f64 = 0.01; // 1% random exploration

// ============================================================================
// THEME COLORS
// ============================================================================
fn get_tile_color(value: u32) -> Color {
    match value {
        0 => Color::from_hex(0x3e4042),       // Empty slot
        1 => Color::from_hex(0x66ccff),       // Blue (Cyan)
        2 => Color::from_hex(0xff6666),       // Red (Coral)
        3 => Color::from_hex(0xffffff),       // White
        6 => Color::from_hex(0xf5e6d3),       // Cream
        12 => Color::from_hex(0xffe4b5),      // Moccasin
        24 => Color::from_hex(0xffc87c),      // Light Orange
        48 => Color::from_hex(0xffaa4c),      // Orange
        96 => Color::from_hex(0xff8c42),      // Dark Orange
        192 => Color::from_hex(0xff6b35),     // Burnt Orange
        384 => Color::from_hex(0xe74c3c),     // Red-ish
        768 => Color::from_hex(0xc0392b),     // Dark Red
        1536 => Color::from_hex(0x9b59b6),    // Purple
        3072 => Color::from_hex(0x8e44ad),    // Dark Purple
        6144 => Color::from_hex(0x2980b9),    // Royal Blue
        12288 => Color::from_hex(0x1abc9c),   // Teal
        24576.. => Color::from_hex(0x16a085), // Dark Teal
        _ => Color::from_hex(0xf5f5f5),
    }
}

fn get_text_color(value: u32) -> Color {
    match value {
        0 => Color::from_hex(0x555555),
        1 | 2 => WHITE,
        3..=12 => Color::from_hex(0x333333),
        _ => WHITE,
    }
}

fn get_bg_color() -> Color {
    Color::from_hex(0x1a1a2e)
}

fn get_hint_panel_bg() -> Color {
    Color::from_hex(0x16213e)
}

// ============================================================================
// DRAWING FUNCTIONS
// ============================================================================
fn get_pos_from_index(index: usize) -> (f32, f32) {
    let row = (index / 4) as f32;
    let col = (index % 4) as f32;
    let x = BOARD_OFFSET_X + col * (TILE_SIZE + PADDING);
    let y = BOARD_OFFSET_Y + row * (TILE_SIZE + PADDING);
    (x, y)
}

fn draw_rounded_rect(x: f32, y: f32, w: f32, h: f32, r: f32, color: Color) {
    // Main rectangle areas (cross shape)
    draw_rectangle(x + r, y, w - 2.0 * r, h, color);
    draw_rectangle(x, y + r, w, h - 2.0 * r, color);

    // Corner circles
    draw_circle(x + r, y + r, r, color);
    draw_circle(x + w - r, y + r, r, color);
    draw_circle(x + r, y + h - r, r, color);
    draw_circle(x + w - r, y + h - r, r, color);
}

fn draw_tile(x: f32, y: f32, value: u32, scale: f32) {
    let color = get_tile_color(value);
    let size = TILE_SIZE * scale;
    let offset = (TILE_SIZE - size) / 2.0;
    let corner_radius = 12.0 * scale;

    // Drop shadow
    if value > 0 {
        draw_rounded_rect(
            x + offset + 4.0,
            y + offset + 4.0,
            size,
            size,
            corner_radius,
            Color::new(0.0, 0.0, 0.0, 0.3),
        );
    }

    // Main tile
    draw_rounded_rect(x + offset, y + offset, size, size, corner_radius, color);

    if value == 0 {
        return;
    }

    // Text
    let text = format!("{}", value);
    let text_color = get_text_color(value);

    // Dynamic font size
    let font_size = if value >= 10000 {
        22.0
    } else if value >= 1000 {
        28.0
    } else if value >= 100 {
        34.0
    } else {
        42.0
    };

    let text_dims = measure_text(&text, None, font_size as u16, 1.0);
    draw_text(
        &text,
        x + offset + (size - text_dims.width) / 2.0,
        y + offset + (size + text_dims.height * 0.7) / 2.0,
        font_size,
        text_color,
    );
}

fn draw_hint_tile(x: f32, y: f32, value: u32, size: f32) {
    let color = get_tile_color(value);
    let corner_radius = 6.0;

    draw_rounded_rect(x, y, size, size, corner_radius, color);

    if value > 0 {
        let text = format!("{}", value);
        let font_size = if value >= 100 { 12.0 } else { 16.0 };
        let text_dims = measure_text(&text, None, font_size as u16, 1.0);
        draw_text(
            &text,
            x + (size - text_dims.width) / 2.0,
            y + (size + text_dims.height * 0.7) / 2.0,
            font_size,
            get_text_color(value),
        );
    }
}

fn draw_board(game: &Game) {
    // Draw background grid slots
    for i in 0..16 {
        let (x, y) = get_pos_from_index(i);
        draw_rounded_rect(x, y, TILE_SIZE, TILE_SIZE, 10.0, Color::from_hex(0x2d2d44));
    }

    // Draw tiles
    for row in 0..4 {
        for col in 0..4 {
            let val = game.board[row][col].value;
            let (x, y) = get_pos_from_index(row * 4 + col);
            if val > 0 {
                draw_tile(x, y, val, 1.0);
            }
        }
    }
}

fn draw_header(game: &Game, ai_enabled: bool, speed: f32) {
    // Title with gradient effect
    let title = "THREES AI";
    draw_text(title, 42.0, 50.0, 48.0, Color::from_hex(0xe94560));

    // Score panel
    let score_text = format!("Score: {:.0}", game.score);
    draw_text(&score_text, 42.0, 85.0, 28.0, Color::from_hex(0xf39c12));

    // AI status indicator
    let ai_status = if ai_enabled { "AI: ON" } else { "AI: OFF" };
    let ai_color = if ai_enabled {
        Color::from_hex(0x2ecc71)
    } else {
        Color::from_hex(0xe74c3c)
    };
    draw_text(ai_status, 280.0, 50.0, 24.0, ai_color);

    // Speed indicator
    let speed_text = format!("Speed: {:.1}x", speed);
    draw_text(&speed_text, 280.0, 80.0, 20.0, Color::from_hex(0x95a5a6));

    // Max tile
    let max_tile = game.get_highest_tile_value();
    let max_text = format!("Max: {}", max_tile);
    draw_text(&max_text, 400.0, 50.0, 24.0, Color::from_hex(0x9b59b6));

    // Move count
    let move_text = format!("Moves: {}", game.num_move);
    draw_text(&move_text, 400.0, 80.0, 20.0, Color::from_hex(0x95a5a6));
}

fn draw_hints(hints: &[u32]) {
    let panel_x = BOARD_OFFSET_X + 4.0 * (TILE_SIZE + PADDING) + 20.0;
    let panel_y = BOARD_OFFSET_Y;
    let panel_width = 120.0;
    let panel_height = 180.0;

    // Panel background
    draw_rounded_rect(
        panel_x,
        panel_y,
        panel_width,
        panel_height,
        10.0,
        get_hint_panel_bg(),
    );

    // Title
    draw_text(
        "NEXT",
        panel_x + 35.0,
        panel_y + 30.0,
        20.0,
        Color::from_hex(0x7f8c8d),
    );

    // Hint tiles
    let hint_size = 40.0;
    let start_y = panel_y + 50.0;
    for (i, &hint) in hints.iter().enumerate() {
        let x = panel_x + (panel_width - hint_size) / 2.0;
        let y = start_y + i as f32 * (hint_size + 10.0);
        draw_hint_tile(x, y, hint, hint_size);
    }
}

fn draw_controls_help() {
    let board_width = 4.0 * (TILE_SIZE + PADDING) - PADDING;
    let start_y = BOARD_OFFSET_Y + board_width + 30.0;

    let controls = [
        "[SPACE]  Toggle AI",
        "[R]  Reset Game",
        "[+/-]  Speed",
        "[Arrow/WASD]  Manual Play",
    ];

    for (i, text) in controls.iter().enumerate() {
        draw_text(
            text,
            BOARD_OFFSET_X,
            start_y + i as f32 * 24.0,
            18.0,
            Color::from_hex(0x7f8c8d),
        );
    }
}

fn draw_game_over_overlay(score: f64, max_tile: u32) {
    // Semi-transparent overlay
    draw_rectangle(
        0.0,
        0.0,
        screen_width(),
        screen_height(),
        Color::new(0.0, 0.0, 0.0, 0.7),
    );

    // Game over box
    let box_width = 300.0;
    let box_height = 200.0;
    let box_x = (screen_width() - box_width) / 2.0;
    let box_y = (screen_height() - box_height) / 2.0;

    draw_rounded_rect(
        box_x,
        box_y,
        box_width,
        box_height,
        15.0,
        Color::from_hex(0x2d3436),
    );

    // Game over text
    draw_text(
        "GAME OVER",
        box_x + 55.0,
        box_y + 60.0,
        36.0,
        Color::from_hex(0xe74c3c),
    );

    // Score
    let score_text = format!("Score: {:.0}", score);
    draw_text(
        &score_text,
        box_x + 80.0,
        box_y + 100.0,
        28.0,
        Color::from_hex(0xf39c12),
    );

    // Max tile
    let max_text = format!("Max Tile: {}", max_tile);
    draw_text(
        &max_text,
        box_x + 70.0,
        box_y + 135.0,
        24.0,
        Color::from_hex(0x9b59b6),
    );

    // Restart hint
    draw_text(
        "[R] to Restart",
        box_x + 85.0,
        box_y + 175.0,
        20.0,
        Color::from_hex(0x95a5a6),
    );
}

// ============================================================================
// AI LOGIC
// ============================================================================
fn get_ai_action(env: &ThreesEnv, brain: &NTupleNetwork, epsilon: f64) -> u32 {
    // Use macroquad's built-in rand for epsilon check
    let random_val: f64 = rand::gen_range(0.0, 1.0);

    // Epsilon-greedy: with probability epsilon, pick random action
    if random_val < epsilon {
        env.get_random_valid_action()
    } else {
        // Safe policy (minimax): pick action that maximizes the minimum future value
        env.get_best_action_safe(brain)
    }
}

fn direction_to_action(dir: Direction) -> u32 {
    match dir {
        Direction::Up => 0,
        Direction::Down => 1,
        Direction::Left => 2,
        Direction::Right => 3,
    }
}

// ============================================================================
// MAIN
// ============================================================================
fn window_conf() -> Conf {
    Conf {
        window_title: "Threes AI - Safe Policy Demo".to_string(),
        window_width: 640,
        window_height: 680,
        window_resizable: false,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    // Try to load brain
    let brain = match NTupleNetwork::load_from_msgpack("brain_ep_12230000.msgpack") {
        Ok(b) => {
            println!("✅ Loaded brain successfully!");
            Some(b)
        }
        Err(e) => {
            eprintln!("⚠️  Could not load brain: {}. AI will use random moves.", e);
            None
        }
    };

    // Initialize game environment
    let mut env = ThreesEnv::new(0.99);
    env.reset();

    // Game state
    let mut ai_enabled = true;
    let mut ai_speed = 2.0_f32; // moves per second
    let mut last_ai_move_time = get_time();

    // Animation state
    let mut pop_scale = 1.0_f32;

    loop {
        clear_background(get_bg_color());

        // --- INPUT HANDLING ---
        if is_key_pressed(KeyCode::Space) {
            ai_enabled = !ai_enabled;
        }

        if is_key_pressed(KeyCode::R) {
            env.reset();
            pop_scale = 1.0;
        }

        // Speed control
        if is_key_pressed(KeyCode::Equal) || is_key_pressed(KeyCode::KpAdd) {
            ai_speed = (ai_speed * 1.5).min(20.0);
        }
        if is_key_pressed(KeyCode::Minus) || is_key_pressed(KeyCode::KpSubtract) {
            ai_speed = (ai_speed / 1.5).max(0.5);
        }

        // Manual controls (when AI is off or for override)
        let manual_action = if is_key_pressed(KeyCode::Up) || is_key_pressed(KeyCode::W) {
            Some(0u32)
        } else if is_key_pressed(KeyCode::Down) || is_key_pressed(KeyCode::S) {
            Some(1u32)
        } else if is_key_pressed(KeyCode::Left) || is_key_pressed(KeyCode::A) {
            Some(2u32)
        } else if is_key_pressed(KeyCode::Right) || is_key_pressed(KeyCode::D) {
            Some(3u32)
        } else {
            None
        };

        // --- GAME LOGIC ---
        let game_over = env.game.game_over;

        if !game_over {
            let mut should_move = false;
            let mut action = 0u32;

            // Manual input takes priority
            if let Some(manual_act) = manual_action {
                let dir = match manual_act {
                    0 => Direction::Up,
                    1 => Direction::Down,
                    2 => Direction::Left,
                    3 => Direction::Right,
                    _ => Direction::Up,
                };
                if env.game.can_move(dir) {
                    action = manual_act;
                    should_move = true;
                }
            } else if ai_enabled {
                // AI move based on speed
                let current_time = get_time();
                if current_time - last_ai_move_time >= (1.0 / ai_speed as f64) {
                    action = if let Some(ref b) = brain {
                        get_ai_action(&env, b, EPSILON)
                    } else {
                        env.get_random_valid_action()
                    };
                    should_move = true;
                    last_ai_move_time = current_time;
                }
            }

            if should_move {
                let dir = match action {
                    0 => Direction::Up,
                    1 => Direction::Down,
                    2 => Direction::Left,
                    3 => Direction::Right,
                    _ => Direction::Up,
                };

                if env.game.can_move(dir) {
                    env.game.move_dir(dir);
                    env.game.check_game_over();
                    pop_scale = 1.1; // Trigger pop animation
                }
            }
        }

        // Pop animation decay
        if pop_scale > 1.0 {
            pop_scale = 1.0 + (pop_scale - 1.0) * 0.85;
            if pop_scale < 1.005 {
                pop_scale = 1.0;
            }
        }

        // --- DRAWING ---
        draw_header(&env.game, ai_enabled, ai_speed);
        draw_board(&env.game);
        draw_hints(&env.game.hints);
        draw_controls_help();

        if game_over {
            draw_game_over_overlay(env.game.score, env.game.get_highest_tile_value());
        }

        next_frame().await
    }
}
