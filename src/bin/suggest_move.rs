use linemux::MuxedLines;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration;
use threes_rs::deck_tracker::{get_rank_from_value, DeckTracker};
use threes_rs::game::Direction;
use threes_rs::n_tuple_network::NTupleNetwork;
use threes_rs::threes_env::ThreesEnv;
use threes_rs::tile::Tile;

#[tokio::main]
async fn main() -> std::io::Result<()> {
    // 1. Load Brain
    let brain_path = find_latest_checkpoint().expect("❌ Không tìm thấy file brain_ep_*.msgpack!");
    println!("📂 Loading brain từ: {}", brain_path);
    let brain = NTupleNetwork::load_from_msgpack(&brain_path).expect("Failed to load brain");
    let shared_brain = Arc::new(brain);

    // 2. Log Path
    let log_path = "/home/huy/.local/share/Steam/steamapps/common/Threes/BepInEx/LogOutput.log";
    if !Path::new(log_path).exists() {
        eprintln!("❌ Không tìm thấy file log tại: {}", log_path);
        return Ok(());
    }

    let mut lines = MuxedLines::new()?;
    lines.add_file(log_path).await?;

    println!("🚀 Bot đang lắng nghe game... (Strict Verification Mode)");

    // --- QUẢN LÝ TRẠNG THÁI ---
    let mut persistent_tracker = DeckTracker::new();

    // Đánh dấu move đã xử lý (-1 để bắt được Move 0 ngay từ đầu)
    let mut last_processed_move: i32 = -1;

    // Lưu dự đoán của lượt TRƯỚC để so sánh với kết quả lượt NÀY
    // Dạng: Vec<(Value, Probability)>
    let mut last_turn_predictions: Option<Vec<(u32, f64)>> = None;

    while let Ok(Some(line)) = lines.next_line().await {
        let content = line.line();
        if content.contains("[DATA]") {
            if let Some(data_raw) = content.split("[DATA] ").last() {
                if let Some((board_1d, current_hint, current_moves, current_score)) =
                    parse_log_line(data_raw)
                {
                    let moves_i32 = current_moves as i32;

                    // ==========================================
                    // BƯỚC 1: XỬ LÝ GAME STATE & VERIFICATION
                    // ==========================================

                    // Case A: Game Reset
                    if moves_i32 < last_processed_move {
                        println!(
                            "\n🔄 GAME MỚI (Moves {} < {}) -> RESET TRACKER & DỰ ĐOÁN",
                            current_moves, last_processed_move
                        );
                        persistent_tracker = DeckTracker::new();
                        last_processed_move = -1;
                        last_turn_predictions = None; // Xóa dự đoán cũ vì game mới rồi
                    }

                    // Case B: Phát hiện lượt mới (Board đã thay đổi, Hint mới đã xuất hiện)
                    if moves_i32 > last_processed_move {
                        // --- [QUAN TRỌNG] SO SÁNH DỰ ĐOÁN CŨ VỚI THỰC TẾ ---
                        if let Some(preds) = last_turn_predictions.take() {
                            // Tìm xác suất bot đã gán cho con 'current_hint' này
                            let predicted_prob = preds
                                .iter()
                                .find(|(val, _)| *val == current_hint)
                                .map(|(_, p)| *p)
                                .unwrap_or(0.0);

                            // Kiểm tra xem bot có từng khẳng định 100% cho con khác không?
                            let was_absolute_confidence = preds.iter().any(|(_, p)| *p >= 0.999);

                            if predicted_prob >= 0.999 {
                                println!(
                                    "🎯 TUYỆT ĐỐI (100%): Chính xác. Ra quân {}.",
                                    current_hint
                                );
                            } else if was_absolute_confidence && predicted_prob < 0.999 {
                                // Bot bảo 100% ra con A, nhưng thực tế ra con B -> LỖI
                                println!("❌ SAI NGHIÊM TRỌNG: Chương trình đoán 100% con khác, nhưng lại ra {}!", current_hint);
                            } else if predicted_prob > 0.0 {
                                println!(
                                    "✅ ĐÚNG ({:.1}%): Ra quân {}.",
                                    predicted_prob * 100.0,
                                    current_hint
                                );
                            } else {
                                println!("❌ TRẬT (0%): Ra {} nhưng Bot không lường trước (Tracker lệch).", current_hint);
                            }
                        } else {
                            // Lượt đầu tiên hoặc sau reset, chưa có dự đoán -> Bỏ qua
                        }
                        // ----------------------------------------------------

                        // Update Tracker: Hint hiện tại là con bài đã rút ra khỏi túi
                        persistent_tracker.update(current_hint);

                        last_processed_move = moves_i32;
                    }

                    // ==========================================
                    // BƯỚC 2: KHỞI TẠO ENV & SYNC
                    // ==========================================

                    let mut env = ThreesEnv::new(0.995);

                    // Sync Board & Score
                    let board_2d = map_1d_to_2d(&board_1d);
                    sync_board_state(&mut env.game, board_2d, current_hint, current_score);

                    // Inject Tracker
                    env.game.deck_tracker = persistent_tracker.clone();
                    env.game.num_move = current_moves;

                    // ==========================================
                    // BƯỚC 3: DỰ ĐOÁN TƯƠNG LAI (CHO VÒNG SAU CHECK)
                    // ==========================================

                    let max_value = env.game.get_highest_tile_value();
                    let max_rank = get_rank_from_value(max_value);
                    // Dự đoán: Sau con Hint này, túi bài còn lại những gì?
                    let predictions = env.game.deck_tracker.predict_future(max_rank);

                    // In ra để người dùng xem
                    print!("🔮 Túi bài sắp tới: ");
                    for (val, prob) in &predictions {
                        if *prob > 0.0 {
                            print!("[{}: {:.1}%] ", val, prob * 100.0);
                        }
                    }
                    println!();

                    // Lưu lại dự đoán này để vòng lặp sau đối chiếu
                    last_turn_predictions = Some(predictions);

                    // ==========================================
                    // BƯỚC 4: AI THINK & ACT
                    // ==========================================

                    let (action_opt, val) = env.get_best_action_depth_parallel(&shared_brain, 4);

                    if let Some(action) = action_opt {
                        println!(
                            "🤖 Bot: {:?} | Score: {} | Eval: {:.2}",
                            action, current_score, val
                        );

                        // Giả lập move trong não (để debug nếu cần, không ảnh hưởng logic chính)
                        if env.game.can_move(action) {
                            // env.game.make_full_move(action);
                        }

                        send_key_to_window("steam_app_1818570", action_to_key(Some(action)));

                        // Chờ animation game
                        tokio::time::sleep(Duration::from_millis(150)).await;
                    } else {
                        println!("☠️ Bot bó tay (Game Over).");
                    }
                }
            }
        }
    }
    Ok(())
}

// --- CÁC HÀM HELPER ---

/// Hàm Sync quan trọng: Biến dữ liệu thô từ log thành State chuẩn của Game
fn sync_board_state(
    game: &mut threes_rs::game::Game,
    board_2d: [[u32; 4]; 4],
    next_hint: u32,
    score: u32,
) {
    // 1. Sync Board Tiles
    for r in 0..4 {
        for c in 0..4 {
            game.board[r][c] = Tile::new(board_2d[r][c]);
        }
    }

    // 2. Sync Future Value (Hint)
    game.future_value = next_hint;

    // 3. Sync Score
    // game.score = score as f64;

    // 4. Update Hints vector (Quan trọng cho AI search)
    // Ở Turn hiện tại (Real game), Hints chỉ có duy nhất 1 con số là con đang hiện trên màn hình
    game.hints = vec![next_hint];

    // Lưu ý: DeckTracker không sync ở đây mà được inject từ bên ngoài vào
}

fn parse_log_line(raw: &str) -> Option<(Vec<u32>, u32, u32, u32)> {
    let parts: Vec<&str> = raw.split('|').collect();
    if parts.len() < 4 {
        return None;
    }
    let board: Vec<u32> = parts[0]
        .split(',')
        .map(|s| s.trim().parse().unwrap_or(0))
        .collect();
    let next = parts[1].parse().unwrap_or(0);
    let moves = parts[2].parse().unwrap_or(0);
    let score = parts[3].parse().unwrap_or(0);
    Some((board, next, moves, score))
}

fn map_1d_to_2d(v: &[u32]) -> [[u32; 4]; 4] {
    let mut board = [[0u32; 4]; 4];
    for i in 0..16 {
        let x = i % 4;
        let y = 3 - (i / 4);
        board[y][x] = v[i];
    }
    board
}

fn send_key_to_window(window_class: &str, key: &str) {
    let search = Command::new("xdotool")
        .args(["search", "--onlyvisible", "--class", window_class])
        .output();
    if let Ok(out) = search {
        let s = String::from_utf8_lossy(&out.stdout);
        if let Some(id) = s.lines().last() {
            let _ = Command::new("xdotool")
                .args(["key", "--window", id, "--delay", "40", key])
                .spawn();
        }
    }
}

fn action_to_key(action: Option<Direction>) -> &'static str {
    match action {
        Some(Direction::Up) => "Up",
        Some(Direction::Down) => "Down",
        Some(Direction::Left) => "Left",
        Some(Direction::Right) => "Right",
        _ => "Return",
    }
}

fn find_latest_checkpoint() -> Option<String> {
    let mut max_ep = 0;
    let mut best = None;
    if let Ok(entries) = std::fs::read_dir(".") {
        for entry in entries.flatten() {
            let name = entry.file_name().into_string().unwrap_or_default();
            if name.starts_with("brain_ep_") && name.ends_with(".msgpack") {
                if let Ok(ep) = name
                    .replace("brain_ep_", "")
                    .replace(".msgpack", "")
                    .parse::<u32>()
                {
                    if ep >= max_ep {
                        max_ep = ep;
                        best = Some(name);
                    }
                }
            }
        }
    }
    best
}
