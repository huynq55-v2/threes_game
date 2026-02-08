use linemux::MuxedLines;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::Arc;
use std::time::Duration; // Thêm thư viện này
use threes_rs::game::Direction;
use threes_rs::n_tuple_network::NTupleNetwork;
use threes_rs::threes_env::ThreesEnv;

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

    println!("🚀 Bot đang lắng nghe game... (Bấm New Game để bắt đầu)");

    let mut predicted_next_states: Vec<[[u32; 4]; 4]> = Vec::new();

    while let Ok(Some(line)) = lines.next_line().await {
        let content = line.line();
        if content.contains("[DATA]") {
            if let Some(data_raw) = content.split("[DATA] ").last() {
                if let Some((board_1d, next_tile, moves, score)) = parse_log_line(data_raw) {
                    // --- BƯỚC 1: KIỂM CHỨNG (VALIDATION) ---
                    let current_actual_board = map_1d_to_2d(&board_1d);

                    if !predicted_next_states.is_empty() {
                        let is_valid = predicted_next_states
                            .iter()
                            .any(|s| s == &current_actual_board);

                        if !is_valid {
                            println!("❌ LỖI LOGIC NGHIÊM TRỌNG!");
                            println!("Game thật: {:?}", current_actual_board);
                            println!("Các trạng thái AI mong đợi: {:?}", predicted_next_states);
                            println!("Dừng chương trình để Huy kiểm tra lại Simulator.");
                            std::process::exit(1); // Thoát ngay lập tức
                        } else {
                            println!("✅ Trạng thái khớp với Simulator.");
                        }
                    }

                    // --- BƯỚC 2: TÍNH TOÁN NƯỚC ĐI TIẾP THEO ---
                    let mut env = ThreesEnv::new(0.995);
                    sync_board(&mut env.game, map_1d_to_2d(&board_1d));
                    env.game.future_value = next_tile;

                    let (action, _value) = env.get_best_action_ply(&shared_brain, 7);
                    let dir = match action {
                        0 => Direction::Up,
                        1 => Direction::Down,
                        2 => Direction::Left,
                        3 => Direction::Right,
                        _ => unreachable!(),
                    };

                    // LƯU DỰ ĐOÁN CHO BƯỚC SAU
                    predicted_next_states = env.game.simulate_move(dir);

                    // Gửi phím
                    send_key_to_window("steam_app_1818570", action_to_key(action));
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            }
        }
    }
    Ok(())
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

fn find_latest_checkpoint() -> Option<String> {
    let mut max_ep = 0;
    let mut best_path = None;
    if let Ok(entries) = fs::read_dir(".") {
        for entry in entries.flatten() {
            let name = entry.file_name().into_string().unwrap_or_default();
            if name.starts_with("brain_ep_") && name.ends_with(".msgpack") {
                let ep = name
                    .replace("brain_ep_", "")
                    .replace(".msgpack", "")
                    .parse::<u32>()
                    .unwrap_or(0);
                if ep >= max_ep {
                    max_ep = ep;
                    best_path = Some(name);
                }
            }
        }
    }
    best_path
}

// --- HÀM GỬI PHÍM ĐÃ NÂNG CẤP ---
fn send_key_to_window(window_class: &str, key: &str) {
    // 1. Tìm ID cửa sổ THỰC SỰ (Chỉ tìm cửa sổ hiện hình --onlyvisible)
    // Steam hay tạo cửa sổ ẩn, nếu gửi vào đó sẽ tạch.
    let search_output = Command::new("xdotool")
        .args(["search", "--onlyvisible", "--class", window_class])
        .output();

    if let Ok(output) = search_output {
        let ids_str = String::from_utf8_lossy(&output.stdout);
        // Lấy ID cuối cùng trong danh sách (thường là cửa sổ game active sau cùng)
        if let Some(window_id) = ids_str.lines().last() {
            // 2. Gửi phím với độ trễ (Delay)
            // --delay 100: Giữ phím 100ms. Unity cần cái này để nhận diện input chắc chắn.
            let _ = Command::new("xdotool")
                .args(["key", "--window", window_id, "--delay", "40", key])
                .spawn();
        } else {
            eprintln!("⚠️ Không tìm thấy cửa sổ game hiển thị (visible)!");
        }
    } else {
        eprintln!("❌ Lỗi gọi xdotool search");
    }
}

// 1. Chuyển mảng 1D từ Log (Dưới lên trên) thành mảng 2D (Trên xuống dưới)
fn map_1d_to_2d(v: &[u32]) -> [[u32; 4]; 4] {
    let mut board = [[0u32; 4]; 4];
    for i in 0..16 {
        let x = i % 4;
        let y = 3 - (i / 4); // Đảo trục Y: 0-3 thành hàng 3, 12-15 thành hàng 0
        board[y][x] = v[i];
    }
    board
}

// 2. Chuyển ID action của AI thành tên phím cho xdotool
fn action_to_key(action: u32) -> &'static str {
    match action {
        0 => "Up",
        1 => "Down",
        2 => "Left",
        3 => "Right",
        _ => "Up", // Default
    }
}

// 3. Vì Game của Huy chưa có set_board, ta gán thủ công vào từng Tile
// Huy kiểm tra xem env.game.board hay env.game.grid nhé (dựa trên lỗi trước là board)
fn sync_board(game: &mut threes_rs::game::Game, source: [[u32; 4]; 4]) {
    for y in 0..4 {
        for x in 0..4 {
            game.board[y][x].value = source[y][x];
        }
    }
}
