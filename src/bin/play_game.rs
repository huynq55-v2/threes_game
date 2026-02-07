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
const EPSILON: f64 = 0.0; // 1% random exploration
const ANIMATION_SPEED: f32 = 5.0; // Speed of slide animation

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
// STRUCTS FOR TRANSITION & ANIMATION
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActionType {
    Slide,  // Tile moves from A to B
    Merge,  // Tile merges into another (pop effect)
    Spawn,  // New tile appears
    Static, // No movement
}

#[derive(Clone, Copy, Debug)]
pub struct TileEvent {
    pub from_index: usize, // 0..15
    pub to_index: usize,   // 0..15
    pub action: ActionType,
    pub value: u32,
    pub merged_value: Option<u32>, // If action is merge, this is the result value
}

#[derive(Clone)]
pub struct RenderState {
    pub grid: [[u32; 4]; 4],
    pub next_hints: Vec<u32>,
    pub score: f64,
}

impl RenderState {
    fn from_game(game: &Game) -> Self {
        let mut grid = [[0u32; 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                grid[r][c] = game.board[r][c].value;
            }
        }
        Self {
            grid,
            next_hints: game.hints.clone(),
            score: game.score,
        }
    }
}

pub struct Transition {
    pub events: Vec<TileEvent>,
    pub end_state: RenderState,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct StepData {
    direction: usize,
    board: [[u32; 4]; 4],
    score: f64,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct GameReplay {
    score: f64,
    max_tile: u32,
    initial_board: [[u32; 4]; 4],
    steps: Vec<StepData>,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct StepDataLocal {
    // avoid conflict if needed, but sticking to identical names
    // ...
}

// ============================================================================
// TRANSITION LOGIC
// ============================================================================

/// Determines rotation index (0..3) based on direction, to normalize shift to "Left".
fn get_rotations_needed(dir: Direction) -> u8 {
    match dir {
        Direction::Left => 0,
        Direction::Down => 1,
        Direction::Right => 2,
        Direction::Up => 3,
    }
}

/// Maps logical (r,c) on rotated board to physical board index (0..15).
fn map_rotated_index(r: usize, c: usize, rot: u8) -> usize {
    let (real_r, real_c) = match rot % 4 {
        0 => (r, c),
        1 => (3 - c, r),
        2 => (3 - r, 3 - c),
        3 => (c, 3 - r),
        _ => unreachable!(),
    };
    real_r * 4 + real_c
}

/// Computes the events that happen during a move.
/// This mimics `threes_rs::game::Game::shift_board_left` but records events.
fn calculate_transition(game: &Game, dir: Direction) -> Transition {
    let mut events = Vec::new();
    let rot = get_rotations_needed(dir);

    // Track which physical indices have been processed/moved
    // We map 4x4 flags.
    let mut moved_flags = [false; 16];

    for r in 0..4 {
        // We scan each row (in rotated space)
        // Similar to process_single_row in game.rs
        // Logic: Find first move/merge pair from left (0..1, 1..2, 2..3)
        // If found, execute it and shift the rest.

        let mut shift_happened = false;

        for c in 0..3 {
            let target_idx = map_rotated_index(r, c, rot);
            let source_idx = map_rotated_index(r, c + 1, rot);

            let target_val = game.board[target_idx / 4][target_idx % 4].value;
            let source_val = game.board[source_idx / 4][source_idx % 4].value;

            if source_val == 0 {
                continue;
            }

            // Check merge/move condition
            // 1. Move to empty
            // 2. Merge 1+2=3
            // 3. Merge X+X -> 2X (X>=3)

            let (new_val, is_merge) = if target_val == 0 {
                (source_val, false)
            } else if target_val + source_val == 3 {
                (3, true)
            } else if target_val >= 3 && target_val == source_val {
                (target_val * 2, true)
            } else {
                continue; // Next pair
            };

            // Found action!
            shift_happened = true;

            // 1. Event for the source tile moving to target
            events.push(TileEvent {
                from_index: source_idx,
                to_index: target_idx,
                action: if is_merge {
                    ActionType::Merge
                } else {
                    ActionType::Slide
                },
                value: source_val,
                merged_value: if is_merge { Some(new_val) } else { None },
            });
            moved_flags[source_idx] = true;

            // If it was a merge, the target tile technically "stays" there and waits for the incoming tile
            // to merge into it. Or we can just say target is static (it's overwritten at end).
            // Visually, the source slides INTO target. Target stays put until hit.
            if target_val > 0 {
                events.push(TileEvent {
                    from_index: target_idx,
                    to_index: target_idx,
                    action: ActionType::Static,
                    value: target_val,
                    merged_value: None,
                });
                moved_flags[target_idx] = true;
            }

            // 2. Shift the rest of the row (c+2 .. 3) moves Left by 1
            for k in (c + 1)..3 {
                let from_k_idx = map_rotated_index(r, k + 1, rot);
                let to_k_idx = map_rotated_index(r, k, rot);
                let val_k = game.board[from_k_idx / 4][from_k_idx % 4].value;

                if val_k > 0 {
                    events.push(TileEvent {
                        from_index: from_k_idx,
                        to_index: to_k_idx,
                        action: ActionType::Slide,
                        value: val_k,
                        merged_value: None,
                    });
                    moved_flags[from_k_idx] = true;
                }
            }

            break; // Threes only does one operation per row per turn
        }

        // If no shift happened in this row, mark existing tiles as Static
        if !shift_happened {
            // But we need to be careful not to mark tiles we already added?
            // No, we loop per r.
        }
    }

    // Add Static events for tiles that didn't move
    for i in 0..16 {
        if !moved_flags[i] {
            let r = i / 4;
            let c = i % 4;
            let val = game.board[r][c].value;
            if val > 0 {
                events.push(TileEvent {
                    from_index: i,
                    to_index: i,
                    action: ActionType::Static,
                    value: val,
                    merged_value: None, // Will stay same
                });
            }
        }
    }

    // Determine Spawn is tricky here because calculate_transition doesn't know RANDOM spawn.
    // Use an empty end_state for now; caller will fill it after executing move.

    // Create dummy RenderState; will be updated by caller
    let dummy_state = RenderState {
        grid: [[0; 4]; 4],
        next_hints: vec![],
        score: 0.0,
    };

    Transition {
        events,
        end_state: dummy_state,
    }
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
    draw_rectangle(x + r, y, w - 2.0 * r, h, color);
    draw_rectangle(x, y + r, w, h - 2.0 * r, color);
    draw_circle(x + r, y + r, r, color);
    draw_circle(x + w - r, y + r, r, color);
    draw_circle(x + r, y + h - r, r, color);
    draw_circle(x + w - r, y + h - r, r, color);
}

fn draw_tile(x: f32, y: f32, value: u32, scale: f32, alpha: f32) {
    let mut color = get_tile_color(value);
    color.a = alpha;

    let size = TILE_SIZE * scale;
    let offset = (TILE_SIZE - size) / 2.0;
    let corner_radius = 12.0 * scale;

    // Drop shadow
    if value > 0 {
        draw_rounded_rect(
            x + offset + 4.0 * scale,
            y + offset + 4.0 * scale,
            size,
            size,
            corner_radius,
            Color::new(0.0, 0.0, 0.0, 0.3 * alpha),
        );
    }

    // Main tile
    draw_rounded_rect(x + offset, y + offset, size, size, corner_radius, color);

    if value == 0 {
        return;
    }

    // Text
    let text = format!("{}", value);
    let mut text_color = get_text_color(value);
    text_color.a = alpha;

    // Dynamic font size
    let base_font_size = if value >= 10000 {
        22.0
    } else if value >= 1000 {
        28.0
    } else if value >= 100 {
        34.0
    } else {
        42.0
    };
    let font_size = base_font_size * scale;

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

fn draw_header_info(score: f64, max_tile: u32, moves: u32, ai_enabled: bool, speed: f32) {
    // Title
    draw_text("THREES AI", 42.0, 50.0, 48.0, Color::from_hex(0xe94560));

    // Score
    draw_text(
        &format!("Score: {:.0}", score),
        42.0,
        85.0,
        28.0,
        Color::from_hex(0xf39c12),
    );

    // AI Status
    let ai_status = if ai_enabled { "AI: ON" } else { "AI: OFF" };
    let ai_color = if ai_enabled {
        Color::from_hex(0x2ecc71)
    } else {
        Color::from_hex(0xe74c3c)
    };
    draw_text(ai_status, 280.0, 50.0, 24.0, ai_color);

    // Speed
    draw_text(
        &format!("Speed: {:.1}x", speed),
        280.0,
        80.0,
        20.0,
        Color::from_hex(0x95a5a6),
    );

    // Stats
    draw_text(
        &format!("Max: {}", max_tile),
        400.0,
        50.0,
        24.0,
        Color::from_hex(0x9b59b6),
    );
    draw_text(
        &format!("Moves: {}", moves),
        400.0,
        80.0,
        20.0,
        Color::from_hex(0x95a5a6),
    );
}

fn draw_hints(hints: &[u32]) {
    let panel_x = BOARD_OFFSET_X + 4.0 * (TILE_SIZE + PADDING) + 20.0;
    let panel_y = BOARD_OFFSET_Y;
    let panel_width = 120.0;

    // Panel background
    draw_rounded_rect(
        panel_x,
        panel_y,
        panel_width,
        180.0,
        10.0,
        get_hint_panel_bg(),
    );

    draw_text(
        "NEXT",
        panel_x + 35.0,
        panel_y + 30.0,
        20.0,
        Color::from_hex(0x7f8c8d),
    );

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
        "[SPACE] Toggle AI",
        "[R] Reset",
        "[+/-] Speed",
        "[Arrow/WASD] Play",
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
    draw_rectangle(
        0.0,
        0.0,
        screen_width(),
        screen_height(),
        Color::new(0.0, 0.0, 0.0, 0.7),
    );
    let (w, h) = (300.0, 200.0);
    let (x, y) = ((screen_width() - w) / 2.0, (screen_height() - h) / 2.0);
    draw_rounded_rect(x, y, w, h, 15.0, Color::from_hex(0x2d3436));
    draw_text(
        "GAME OVER",
        x + 55.0,
        y + 60.0,
        36.0,
        Color::from_hex(0xe74c3c),
    );
    draw_text(
        &format!("Score: {:.0}", score),
        x + 80.0,
        y + 100.0,
        28.0,
        Color::from_hex(0xf39c12),
    );
    draw_text(
        &format!("Max Tile: {}", max_tile),
        x + 70.0,
        y + 135.0,
        24.0,
        Color::from_hex(0x9b59b6),
    );
    draw_text(
        "[R] to Restart",
        x + 85.0,
        y + 175.0,
        20.0,
        Color::from_hex(0x95a5a6),
    );
}

// ============================================================================
// AI LOGIC
// ============================================================================
fn get_ai_action(env: &ThreesEnv, brain: &NTupleNetwork, epsilon: f64) -> u32 {
    let random_val: f64 = rand::gen_range(0.0, 1.0);
    if random_val < epsilon {
        env.get_random_valid_action()
    } else {
        env.get_best_action_recursive(brain)
    }
}

// ============================================================================
// MAIN
// ============================================================================
fn window_conf() -> Conf {
    Conf {
        window_title: "Threes AI - Smooth Transition".to_string(),
        window_width: 640,
        window_height: 680,
        window_resizable: false,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let brain = match NTupleNetwork::load_from_msgpack("brain_ep_4320000.msgpack") {
        Ok(b) => {
            println!("✅ Loaded brain successfully!");
            Some(b)
        }
        Err(_) => {
            println!("⚠️ Could not load brain.");
            None
        }
    };

    let mut env = ThreesEnv::new(0.995);
    env.reset();

    let mut ai_enabled = true;
    let mut ai_speed = 2.0_f32;
    let mut last_ai_move_time = get_time();

    // TRANSITION STATE
    let mut current_transition: Option<Transition> = None;
    let mut animation_t: f32 = 1.0; // 0.0 to 1.0 = sliding; > 1.0 = static/pop

    // Previous state snapshot (for static display if needed, but we assume we move to target)
    let mut render_state = RenderState::from_game(&env.game);

    // Replay state
    let mut replay_data: Option<GameReplay> = None;
    let mut is_replay_mode = false;
    let mut replay_step = 0;
    let mut replay_auto_play = false;
    let mut last_replay_step_time = get_time();

    loop {
        clear_background(get_bg_color());

        if is_key_pressed(KeyCode::L) {
            // Try load replay
            if let Ok(content) = std::fs::read_to_string("best_replay.json") {
                if let Ok(parsed) = serde_json::from_str::<GameReplay>(&content) {
                    replay_data = Some(parsed);
                    is_replay_mode = true;
                    replay_step = 0;
                    replay_auto_play = true;

                    // Init replay state
                    if let Some(ref r) = replay_data {
                        env.game.score = 0.0; // Start
                                              // Set board to initial
                        for row in 0..4 {
                            for col in 0..4 {
                                env.game.board[row][col].value = r.initial_board[row][col];
                            }
                        }
                        render_state = RenderState::from_game(&env.game); // Update visual
                    }
                    println!("Loaded Replay!");
                } else {
                    println!("Failed to parse best_replay.json");
                }
            } else {
                println!("Could not read best_replay.json");
            }
        }

        if is_replay_mode {
            // REPLAY LOGIC
            if is_key_pressed(KeyCode::R) {
                is_replay_mode = false;
                replay_data = None;
                env.reset();
                render_state = RenderState::from_game(&env.game);
            }
            if is_key_pressed(KeyCode::Space) {
                replay_auto_play = !replay_auto_play;
            }

            let mut step_change = 0;
            if is_key_pressed(KeyCode::Right) || is_key_pressed(KeyCode::D) {
                step_change = 1;
                replay_auto_play = false;
            }
            if is_key_pressed(KeyCode::Left) || is_key_pressed(KeyCode::A) {
                step_change = -1;
                replay_auto_play = false;
            }

            if replay_auto_play {
                if get_time() - last_replay_step_time > 0.3 {
                    // 300ms per step
                    step_change = 1;
                    last_replay_step_time = get_time();
                }
            }

            if step_change != 0 {
                if let Some(ref r) = replay_data {
                    let new_step =
                        (replay_step as i32 + step_change).clamp(0, r.steps.len() as i32) as usize;

                    if new_step != replay_step {
                        replay_step = new_step;

                        // Apply state
                        if replay_step == 0 {
                            // Initial
                            for row in 0..4 {
                                for col in 0..4 {
                                    env.game.board[row][col].value = r.initial_board[row][col];
                                }
                            }
                            env.game.score = 0.0;
                        } else {
                            let step_idx = replay_step - 1;
                            let step = &r.steps[step_idx];
                            for row in 0..4 {
                                for col in 0..4 {
                                    env.game.board[row][col].value = step.board[row][col];
                                }
                            }
                            env.game.score = step.score;
                        }

                        // Update visual immediately (no smooth transition for replay yet)
                        render_state = RenderState::from_game(&env.game);
                        animation_t = 100.0; // No animation for replay skipping
                    }
                }
            }
        } else {
            // NORMAL GAMEPLAY LOGIC
            if is_key_pressed(KeyCode::Space) {
                ai_enabled = !ai_enabled;
            }
            if is_key_pressed(KeyCode::R) {
                env.reset();
                animation_t = 100.0;
                current_transition = None;
                render_state = RenderState::from_game(&env.game);
            }
            if is_key_pressed(KeyCode::Equal) {
                ai_speed = (ai_speed * 1.5).min(20.0);
            }
            if is_key_pressed(KeyCode::Minus) {
                ai_speed = (ai_speed / 1.5).max(0.5);
            }

            let manual_action = if is_key_pressed(KeyCode::Up) || is_key_pressed(KeyCode::W) {
                Some(0)
            } else if is_key_pressed(KeyCode::Down) || is_key_pressed(KeyCode::S) {
                Some(1)
            } else if is_key_pressed(KeyCode::Left) || is_key_pressed(KeyCode::A) {
                Some(2)
            } else if is_key_pressed(KeyCode::Right) || is_key_pressed(KeyCode::D) {
                Some(3)
            } else {
                None
            };

            // Process Move if not animating ...
            let can_input = animation_t >= 1.0 && !env.game.game_over;

            if can_input {
                let mut action = None;

                if let Some(act) = manual_action {
                    action = Some(act);
                } else if ai_enabled {
                    let current_time = get_time();
                    if current_time - last_ai_move_time >= (1.0 / ai_speed as f64) {
                        if let Some(ref b) = brain {
                            action = Some(get_ai_action(&env, b, EPSILON));
                        } else {
                            action = Some(env.get_random_valid_action());
                        }
                        last_ai_move_time = current_time;
                    }
                }

                if let Some(act) = action {
                    let dir = match act {
                        0 => Direction::Up,
                        1 => Direction::Down,
                        2 => Direction::Left,
                        3 => Direction::Right,
                        _ => Direction::Up,
                    };

                    if env.game.can_move(dir) {
                        // 1. Calculate Transition Events
                        let mut transition = calculate_transition(&env.game, dir);

                        // EXECUTE MOVE
                        env.game.move_dir(dir);
                        env.game.check_game_over();

                        // FIND SPAWN Logic...
                        let mut predicted_grid = [[0u32; 4]; 4];

                        for event in &transition.events {
                            match event.action {
                                ActionType::Merge => {
                                    let (r, c) = (event.to_index / 4, event.to_index % 4);
                                    predicted_grid[r][c] =
                                        event.merged_value.unwrap_or(event.value);
                                }
                                ActionType::Slide => {
                                    let (r, c) = (event.to_index / 4, event.to_index % 4);
                                    predicted_grid[r][c] = event.value;
                                }
                                ActionType::Static => {
                                    let (r, c) = (event.to_index / 4, event.to_index % 4);
                                    predicted_grid[r][c] = event.value;
                                }
                                _ => {}
                            }
                        }

                        for r in 0..4 {
                            for c in 0..4 {
                                let actual = env.game.board[r][c].value;
                                let pred = predicted_grid[r][c];
                                if actual != pred {
                                    transition.events.push(TileEvent {
                                        from_index: r * 4 + c,
                                        to_index: r * 4 + c,
                                        action: ActionType::Spawn,
                                        value: actual,
                                        merged_value: None,
                                    });
                                }
                            }
                        }

                        transition.end_state = RenderState::from_game(&env.game);
                        current_transition = Some(transition);
                        render_state = RenderState::from_game(&env.game);

                        animation_t = 0.0;
                    }
                }
            }
        }

        // --- RENDER ---
        if animation_t < 1.0 {
            animation_t += get_frame_time() * ANIMATION_SPEED;
        } else if animation_t > 10.0 {
            // Cap it
        } else {
            animation_t += get_frame_time(); // Keep ticking for pop decay logic
        }

        if is_replay_mode {
            draw_text("REPLAY MODE", 160.0, 30.0, 30.0, Color::from_hex(0x00cec9));
            if let Some(ref r) = replay_data {
                draw_text(
                    &format!("Step: {} / {}", replay_step, r.steps.len()),
                    160.0,
                    60.0,
                    20.0,
                    WHITE,
                );
            }
        }

        draw_header_info(
            render_state.score,
            env.game.get_highest_tile_value(),
            env.game.num_move,
            ai_enabled && !is_replay_mode,
            ai_speed,
        );
        draw_hints(&render_state.next_hints);
        draw_controls_help();

        // DRAW BOARD GRID
        for i in 0..16 {
            let (x, y) = get_pos_from_index(i);
            draw_rounded_rect(x, y, TILE_SIZE, TILE_SIZE, 10.0, Color::from_hex(0x2d2d44));
        }

        // DRAW TILES
        if animation_t < 1.0 {
            if let Some(ref trans) = current_transition {
                // Layer 1: Static and Targets of Merge (Drawing the tile that is being merged INTO)
                // Actually, if A merges into B. B stays static (in visual) until A hits it.
                // Our `calculate_transition` generated a Static event for B?
                // Yes: `if target_val > 0 ... ActionType::Static`.

                for event in &trans.events {
                    if event.action == ActionType::Static {
                        let (x, y) = get_pos_from_index(event.from_index);
                        draw_tile(x, y, event.value, 1.0, 1.0);
                    }
                }

                // Layer 2: Moving items (Slides and Merges)
                for event in &trans.events {
                    if event.action == ActionType::Slide || event.action == ActionType::Merge {
                        let (start_x, start_y) = get_pos_from_index(event.from_index);
                        let (end_x, end_y) = get_pos_from_index(event.to_index);

                        let t = animation_t; // Linear
                                             // Easing optional? let t = t * t * (3.0 - 2.0 * t); // SmoothStep

                        let curr_x = start_x + (end_x - start_x) * t;
                        let curr_y = start_y + (end_y - start_y) * t;

                        draw_tile(curr_x, curr_y, event.value, 1.0, 1.0);
                    }
                }

                // Spawns handled in static phase?
            }
        } else {
            // STATIC / POP PHASE
            // Use current game state (render_state.grid)

            for row in 0..4 {
                for col in 0..4 {
                    let val = render_state.grid[row][col];
                    if val > 0 {
                        let (x, y) = get_pos_from_index(row * 4 + col);
                        let mut scale = 1.0;

                        // Check if this tile was a result of Merge or Spawn for Pop effect
                        if let Some(ref trans) = current_transition {
                            // Find event ending at this index with Merge or Spawn
                            let idx = row * 4 + col;
                            let is_pop_target = trans.events.iter().any(|e| {
                                e.to_index == idx
                                    && (e.action == ActionType::Merge
                                        || e.action == ActionType::Spawn)
                            });

                            if is_pop_target {
                                // Pop effect: 1.0 -> 1.1 -> 1.0
                                // animation_t goes from 1.0 upwards.
                                let dt = animation_t - 1.0;
                                if dt < 0.2 {
                                    // Scale up and down
                                    scale = 1.0 + (0.15 * (1.0 - (dt / 0.2)));
                                }
                            }
                        }

                        draw_tile(x, y, val, scale, 1.0);
                    }
                }
            }
        }

        if env.game.game_over {
            draw_game_over_overlay(env.game.score, env.game.get_highest_tile_value());
        }

        next_frame().await
    }
}
