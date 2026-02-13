use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::sync::Arc;
use std::time::Instant;
use threes_rs::n_tuple_network::NTupleNetwork;
use threes_rs::threes_env::ThreesEnv;

// Cấu trúc lưu thống kê của 1 game
struct GameSpawnStats {
    max_tile: u32,
    // Key: Giá trị quân bài (6, 12, 24...)
    // Value: Số lần quân bài đó được spawn trong suốt ván game
    spawn_history: HashMap<u32, u32>,
}

fn main() {
    // 1. Cấu hình Thread Pool
    let num_threads = 8;
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    println!("🚀 Starting Spawn Distribution Analysis...");

    // 2. Load Brain
    let args: Vec<String> = env::args().collect();
    let brain_path = if args.len() > 1 {
        args[1].clone()
    } else {
        match find_latest_checkpoint() {
            Some(ep) => format!("brain_ep_{}.msgpack", ep),
            None => {
                eprintln!("❌ No brain file found!");
                return;
            }
        }
    };

    println!("📂 Loading brain from: {}", brain_path);
    let brain = NTupleNetwork::load_from_msgpack(&brain_path).expect("Failed to load brain");
    let shared_brain = Arc::new(brain);

    // 3. Số lượng game giả lập
    let total_games = 100_000;
    println!("🎮 Simulating {} games...", total_games);
    let start_time = Instant::now();

    // 4. Chạy giả lập song song
    let results: Vec<GameSpawnStats> = (0..total_games)
        .into_par_iter()
        .map(|_| {
            let mut env = ThreesEnv::new(1.0);
            let brain_ref = shared_brain.clone();

            let mut spawns = HashMap::new();

            // Loop chơi game
            while !env.game.is_game_over() {
                // Lấy nước đi tốt nhất (Depth 1 cho nhanh, Depth 2 cho chính xác)
                let (action, _) = env.get_best_action_depth(&brain_ref, 1);

                if let Some(act) = action {
                    // --- THU THẬP SỐ LIỆU (Trước khi Move) ---
                    // future_value chính là con bài sẽ rớt xuống sau nước đi này
                    let val_to_spawn = resolve_real_spawn_value(env.game.future_value);

                    // Chỉ đếm quân >= 6 (Rank >= 2)
                    if val_to_spawn >= 6 {
                        *spawns.entry(val_to_spawn).or_insert(0) += 1;
                    }

                    // Thực hiện nước đi
                    env.game.make_full_move(act);
                } else {
                    break;
                }
            }

            GameSpawnStats {
                max_tile: env.game.get_highest_tile_value(),
                spawn_history: spawns,
            }
        })
        .collect();

    println!(
        "\n✅ Simulation finished in {:.2}s",
        start_time.elapsed().as_secs_f64()
    );

    // 5. Tổng hợp dữ liệu
    // Map: MaxTile -> List of (Spawn History Maps)
    let mut stats_by_max_tile: BTreeMap<u32, Vec<HashMap<u32, u32>>> = BTreeMap::new();

    for res in results {
        stats_by_max_tile
            .entry(res.max_tile)
            .or_default()
            .push(res.spawn_history);
    }

    println!("\n🧩 SPAWN DISTRIBUTION ANALYSIS (Dropped Tiles)");
    println!("(Percentage of total spawns >= 6 for games ending with specific Max Tile)");
    println!("----------------------------------------------------------------------------------");

    // Duyệt qua từng nhóm Max Tile
    for (&max_tile, game_list) in &stats_by_max_tile {
        // Chỉ quan tâm các game đạt Max Tile >= 48
        if max_tile < 48 {
            continue;
        }

        let num_games = game_list.len();

        // Tổng hợp tất cả các lần spawn trong nhóm này
        let mut total_spawns_map: BTreeMap<u32, u64> = BTreeMap::new();
        let mut grand_total_spawns: u64 = 0;

        for history in game_list {
            for (&val, &count) in history {
                *total_spawns_map.entry(val).or_default() += count as u64;
                grand_total_spawns += count as u64;
            }
        }

        if grand_total_spawns == 0 {
            continue;
        }

        print!("Max Tile {:>4} ({:>4} games): ", max_tile, num_games);

        let mut parts = Vec::new();
        for (&val, &count) in &total_spawns_map {
            // Tính phần trăm: (Số lần spawn quân X / Tổng số lần spawn quân >=6)
            let pct = (count as f64 / grand_total_spawns as f64) * 100.0;

            // Chỉ in nếu có xuất hiện (> 0.0%)
            if pct > 0.0 {
                parts.push(format!("{}: {:.1}%", val, pct));
            }
        }

        println!("{}", parts.join(" | "));
    }
    println!("----------------------------------------------------------------------------------");
}

fn find_latest_checkpoint() -> Option<u32> {
    let mut max_ep = 0;
    let mut found = false;
    if let Ok(entries) = fs::read_dir(".") {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("brain_ep_") && name.ends_with(".msgpack") {
                    let num_part = name
                        .trim_start_matches("brain_ep_")
                        .trim_end_matches(".msgpack");
                    if let Ok(ep) = num_part.parse::<u32>() {
                        if ep >= max_ep {
                            max_ep = ep;
                            found = true;
                        }
                    }
                }
            }
        }
    }
    if found {
        Some(max_ep)
    } else {
        None
    }
}

use rand::Rng;
use std::cmp::max;

// Giả định config chuẩn của Threes
const K_SPECIAL_DEMOTION: u32 = 3;

// Helper: Chuyển từ Rank về Value (Ngược lại của get_rank_from_value)
// Rank 1 -> 3
// Rank 2 -> 6
// Rank 3 -> 12
pub fn get_value_from_rank(rank: u32) -> u32 {
    if rank == 0 {
        return 0; // Hoặc 1, 2 tùy ngữ cảnh, nhưng logic spawn không dùng rank 0
    }
    // Công thức: 3 * 2^(rank - 1)
    3 * (1 << (rank - 1))
}

// Hàm chính: Convert Future Value -> Rank -> Random Downgrade -> Real Value
pub fn resolve_real_spawn_value(future_value: u32) -> u32 {
    // 1. Nếu value <= 3 (Basic Tiles), không có rank để hạ cấp -> Giữ nguyên
    if future_value <= 3 {
        return future_value;
    }

    // 2. Lấy Rank hiện tại
    let rank = get_rank_from_value(future_value);

    // 3. Tính Min Rank (Cận dưới)
    // Logic C#: Mathf.Max(2, rank - settings.kSpecialDemotion + 1)
    // Lưu ý: Rank 2 tương ứng với số 6. Không bao giờ hạ xuống 3 (Rank 1).
    let min_rank = max(2, rank.saturating_sub(K_SPECIAL_DEMOTION) + 1);

    // 4. Tính Max Rank (Cận trên)
    let max_rank = rank;

    // Nếu cận dưới >= cận trên (trường hợp rank thấp), trả về nguyên bản
    if min_rank >= max_rank {
        return future_value;
    }

    // 5. Random trong khoảng [min, max] (Inclusive)
    // Logic C#: Random.Range(min, max + 1) là exclusive max -> tức là lấy [min, max]
    let mut rng = rand::rng();
    let chosen_rank = rng.random_range(min_rank..=max_rank);

    // 6. Convert ngược lại ra Value
    get_value_from_rank(chosen_rank)
}

// Hàm bạn đã cung cấp (giữ nguyên để code chạy được)
pub fn get_rank_from_value(value: u32) -> u32 {
    if value <= 2 {
        return 0;
    }
    if value == 3 {
        return 1;
    }
    (value as f64 / 3.0).log2() as u32 + 1
}
