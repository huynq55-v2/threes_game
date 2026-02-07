use rand::Rng;
use rayon::prelude::*;
use std::fs::{self, File}; // Thêm fs để quét thư mục
use std::io::BufReader;
use std::sync::{Arc, RwLock};
use std::time::Duration;
use std::{env, thread};
use threes_rs::hotload_config::HotLoadConfig;
use threes_rs::{n_tuple_network::NTupleNetwork, pbt::TrainingConfig, threes_env::ThreesEnv};

// Hằng số Tỷ lệ vàng
const GOLDEN_RATIO: f64 = 1.61803398875;

// Struct wrapper pointer (giữ nguyên)
struct SharedBrain {
    network: *mut NTupleNetwork,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum TrainingPolicy {
    Expectimax,
    Safe,
    Afterstate, // New policy
}

unsafe impl Send for SharedBrain {}
unsafe impl Sync for SharedBrain {}

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

fn main() {
    let num_threads = 8;
    let gamma = 0.995;
    let args: Vec<String> = env::args().collect();

    // --- LOGIC 1: TỰ ĐỘNG TÌM FILE SAVE MỚI NHẤT (AUTO-DISCOVERY) ---
    // Nếu người dùng không nhập số, tự động quét thư mục tìm file msgpack có số to nhất.
    let override_episode = find_latest_checkpoint().unwrap_or(0);

    println!("🔎 Start Episode: {}", override_episode);

    // Policy
    let policy_arg = if args.len() > 2 {
        args[2].to_lowercase()
    } else {
        "expect".to_string()
    };

    let training_policy = match policy_arg.as_str() {
        "expect" => {
            println!("🧠 Training Mode: EXPECTIMAX");
            TrainingPolicy::Expectimax
        }
        "safe" => {
            println!("🧠 Training Mode: SAFE");
            TrainingPolicy::Safe
        }
        _ => {
            println!("🧠 Training Mode: AFTERSTATE (Default/Optimized)");
            TrainingPolicy::Afterstate
        }
    };

    let multiplier = args[2].to_lowercase();

    let mut buff_multiplier = 1.0;

    println!("Multiplier Strategy: {}", multiplier);

    // --- SETUP BRAIN ---
    let mut brain = if override_episode > 0 {
        let filename = format!("brain_ep_{}.msgpack", override_episode);
        println!("📂 Loading brain: {}", filename);
        let b = NTupleNetwork::load_from_msgpack(&filename)
            .expect("❌ Không tìm thấy file checkpoint!");
        println!(
            "🧐 LOAD DATA: E={:.1}, S={:.1}, M={:.1}, D={:.1}",
            b.w_empty, b.w_snake, b.w_merge, b.w_disorder
        );
        b
    } else {
        println!("✨ Tạo não mới tinh (Episode 0)...");
        NTupleNetwork::new(0.1, gamma)
    };

    // Logic tương thích ngược cho file cũ
    if override_episode > 0 && brain.total_episodes == 0 {
        println!(
            "⚠️ File cũ chưa có total_episodes, cập nhật thủ công thành {}",
            override_episode
        );
        brain.total_episodes = override_episode;
    }

    // Safety checks
    // if brain.w_empty == 0.0 {
    //     brain.w_empty = 50.0;
    // }
    // if brain.w_snake == 0.0 {
    //     brain.w_snake = 50.0;
    // }
    // if brain.w_merge == 0.0 {
    //     brain.w_merge = 50.0;
    // }
    // if brain.w_disorder == 0.0 {
    //     brain.w_disorder = 50.0;
    // }

    // Config Watcher & PBT
    let hot_config = Arc::new(RwLock::new(HotLoadConfig::default()));
    start_config_watcher(hot_config.clone());
    println!("🔥 Hot Reload ENABLED - Đang theo dõi config.json");
    // let pbt_manager = Arc::new(Mutex::new(PBTManager::new()));

    let chunk_episodes = 80_000;
    let total_target_episodes = 100_000_000;

    // --- CHECKPOINT GỐC (SINGLE SOURCE OF TRUTH) ---
    // Đây là bản chuẩn. Mọi vòng lặp đều clone từ đây ra.
    let mut best_stable_brain = brain.clone();

    // Kỷ lục được tính dựa trên điểm EVAL (evaluation training), không phải điểm train noisy
    let mut best_eval_avg = best_stable_brain.best_overall_avg;

    println!(
        "🚀 Start Training. Baseline Eval Record: {:.2}",
        best_eval_avg
    );
    println!(
        "📊 Current Checkpoint: Ep {} | Config: E={:.1} S={:.1} M={:.1} D={:.1}",
        best_stable_brain.total_episodes,
        best_stable_brain.w_empty,
        best_stable_brain.w_snake,
        best_stable_brain.w_merge,
        best_stable_brain.w_disorder
    );

    // Số lượng game evaluation. 50k để đánh giá kỹ.
    let eval_games = chunk_episodes / 10;

    loop {
        let loop_start = std::time::Instant::now();

        // ============================================================
        // BƯỚC 0: RESET VỀ BẢN CHUẨN ĐỂ TRAIN
        // ============================================================
        brain = best_stable_brain.clone();

        // Tạo pointer MỚI cho vòng lặp này (Quan trọng!)
        // let brain_ptr = SharedBrain {
        //     network: &mut brain as *mut NTupleNetwork,
        // };
        // let shared_brain_loop = Arc::new(brain_ptr);

        // ------------------------------------------------------
        // 1. LOGIC BUFF (Random 1 chỉ số)
        // ------------------------------------------------------
        // let rng = rand::rng();
        // let buff_idx = rng.random_range(0..4);

        // match buff_idx {
        //     0 => {
        //         brain.w_empty *= buff_multiplier;
        //         print!("✨ BUFF EMPTY! ");
        //     }
        //     1 => {
        //         brain.w_snake *= buff_multiplier;
        //         print!("🐍 BUFF SNAKE! ");
        //     }
        //     2 => {
        //         brain.w_merge *= buff_multiplier;
        //         print!("🔗 BUFF MERGE! ");
        //     }
        //     _ => {
        //         brain.w_disorder *= buff_multiplier;
        //         print!("⚡ BUFF DISORDER! ");
        //     }
        // }

        println!(
            "-> Test Config: {:.1}/{:.1}/{:.1}/{:.1}",
            brain.w_empty, brain.w_snake, brain.w_merge, brain.w_disorder
        );

        // Điều chỉnh Phase dựa trên ngưỡng
        if brain.w_empty > 10000.0
            || brain.w_snake > 10000.0
            || brain.w_merge > 10000.0
            || brain.w_disorder > 10000.0
        {
            brain.phase = false; // Chuyển sang giảm
        }

        // if brain.w_empty < 60.0
        //     || brain.w_snake < 60.0
        //     || brain.w_merge < 60.0
        //     || brain.w_disorder < 60.0
        // {
        //     brain.phase = true; // Chuyển sang tăng
        // }

        if brain.phase {
            buff_multiplier = GOLDEN_RATIO;
            print!(" (PHASE: TĂNG 📈) ");
        } else {
            buff_multiplier = 1.0 / GOLDEN_RATIO;
            print!(" (PHASE: GIẢM 📉) ");
        }

        println!("-> Buff Multiplier: {:.2}", buff_multiplier);

        // ============================================================
        // BƯỚC 1: PARALLEL TRAINING - Mỗi thread clone brain riêng
        // Mục đích: Tìm CONFIG tối ưu cho brain hiện tại
        // ============================================================
        println!(
            "🏋️ Training Phase ({} games, {} threads)...",
            chunk_episodes, num_threads
        );

        let ep_per_thread = chunk_episodes as u32 / num_threads;
        let current_base_ep = best_stable_brain.total_episodes;

        // Clone brain gốc để share (read-only reference)
        let base_brain = best_stable_brain.clone();
        let base_config = TrainingConfig {
            w_empty: brain.w_empty,
            w_snake: brain.w_snake,
            w_merge: brain.w_merge,
            w_disorder: brain.w_disorder,
        };

        // Clone hot_config để share vào các thread
        let hot_config_clone = hot_config.clone();

        // Mỗi thread trả về (avg_score, trained_brain, config)
        let thread_results: Vec<(f64, NTupleNetwork, TrainingConfig)> = (0..num_threads)
            .into_par_iter()
            .map(|t_id| {
                // Clone brain RIÊNG cho thread này
                let mut local_brain = base_brain.clone();
                let mut local_env = ThreesEnv::new(gamma);
                let mut rng = rand::rng();

                // Tạo config riêng cho thread này (mutate từ base)
                let mut thread_config = base_config;

                // Mỗi thread mutate ngẫu nhiên 1-2 weights
                for _ in 0..2 {
                    let param_idx = rng.random_range(0..4);
                    let mutate_factor = rng.random_range(1..6) as f64;
                    // tinh buff dua vao buff_multiplier
                    let mutate = buff_multiplier.powf(mutate_factor);

                    match param_idx {
                        0 => thread_config.w_empty *= mutate,
                        1 => thread_config.w_snake *= mutate,
                        2 => thread_config.w_merge *= mutate,
                        _ => thread_config.w_disorder *= mutate,
                    }
                }

                local_brain.w_empty = thread_config.w_empty;
                local_brain.w_snake = thread_config.w_snake;
                local_brain.w_merge = thread_config.w_merge;
                local_brain.w_disorder = thread_config.w_disorder;

                // Train riêng biệt
                let mut total_score = 0.0;
                for local_ep in 0..ep_per_thread {
                    let current_global_ep = (local_ep * num_threads + t_id) + current_base_ep;
                    let progress = current_global_ep as f64 / total_target_episodes as f64;

                    // ============ HOT RELOAD - ĐỌC CONFIG.JSON ============
                    let current_hot = *hot_config_clone.read().unwrap();
                    let mut effective_config = thread_config;

                    if let Some(v) = current_hot.w_empty_override {
                        effective_config.w_empty = v;
                    }
                    if let Some(v) = current_hot.w_snake_override {
                        effective_config.w_snake = v;
                    }
                    if let Some(v) = current_hot.w_merge_override {
                        effective_config.w_merge = v;
                    }
                    if let Some(v) = current_hot.w_disorder_override {
                        effective_config.w_disorder = v;
                    }

                    local_env.set_config(effective_config);

                    // 🔥 QUAN TRỌNG: Cập nhật ngược lại để báo cáo cuối thread và Best Config mang số này!
                    thread_config = effective_config;
                    // ========================================================

                    let mut current_alpha = (0.01 * (1.0 - progress)).max(0.0001);
                    if let Some(v) = current_hot.alpha_override {
                        current_alpha = v;
                    }
                    let mut current_epsilon = (0.2 * (1.0 - (progress / 0.8))).max(0.01);
                    if let Some(v) = current_hot.epsilon_override {
                        current_epsilon = v;
                    }

                    local_env.reset();
                    // local_brain.reset_traces(); // Removed: Traces now managed by env
                    let mut step_count = 0;
                    while !local_env.game.game_over {
                        step_count += 1;
                        if step_count > 20000 {
                            break;
                        }

                        let action = if rng.random_bool(current_epsilon.into()) {
                            local_env.get_random_valid_action()
                        } else {
                            match training_policy {
                                TrainingPolicy::Expectimax => {
                                    local_env.get_best_action_expectimax(&mut local_brain)
                                }
                                TrainingPolicy::Safe => {
                                    local_env.get_best_action_safe(&mut local_brain)
                                }
                                TrainingPolicy::Afterstate => {
                                    local_env.get_best_action_afterstate(&mut local_brain)
                                }
                            }
                        };

                        local_env.train_step(&mut local_brain, action, current_alpha);
                    }

                    total_score += local_env.game.score as f64;

                    // Log progress (chỉ thread 0)
                    if t_id == 0 && local_ep % 2000 == 0 {
                        let running_avg = total_score / (local_ep + 1) as f64;
                        print!(
                            "\r   T0: {:>5}/{} | Avg: {:>5.0} | Cfg: S{:.0} M{:.0}   ",
                            local_ep,
                            ep_per_thread,
                            running_avg,
                            effective_config.w_snake,
                            effective_config.w_merge
                        );
                        use std::io::Write;
                        std::io::stdout().flush().unwrap();
                    }
                }

                let avg_score = total_score / ep_per_thread as f64;
                (avg_score, local_brain, thread_config)
            })
            .collect();

        println!(); // Newline sau progress

        // Tìm best config và merge weights
        let mut best_score = 0.0f64;
        let mut best_config = base_config;

        println!("   Thread Results:");
        for (i, (score, _, cfg)) in thread_results.iter().enumerate() {
            println!(
                "   T{}: Avg={:.0} | E={:.0} S={:.0} M={:.0} D={:.0}",
                i, score, cfg.w_empty, cfg.w_snake, cfg.w_merge, cfg.w_disorder
            );
            if *score > best_score {
                best_score = *score;
                best_config = *cfg;
            }
        }

        // Merge weights bằng Softmax Weighted Averaging (Chuyên nghiệp hơn Linear)
        // Weight_i = exp(Score_i / T) / Sum(exp(Score_j / T))
        // T (Temperature) cao -> Merge đều. T thấp -> Chọn lọc người giỏi nhất.
        let max_score = thread_results
            .iter()
            .map(|(s, _, _)| *s)
            .fold(0.0, f64::max);
        let temperature = max_score * 0.1; // T = 10% của max score

        // Tính mẫu số (denominator) ổn định số học (trừ max_score để tránh overflow exp)
        let mut sum_exp = 0.0;
        let mut softmax_weights = Vec::new();

        for (score, _, _) in &thread_results {
            let val = ((score - max_score) / temperature).exp();
            sum_exp += val;
            softmax_weights.push(val);
        }

        // Normalize weights
        for w in &mut softmax_weights {
            *w /= sum_exp;
        }

        println!(
            "   -> Softmax Merge Weights: {:?}",
            softmax_weights
                .iter()
                .map(|w| (w * 100.0).round() as i32)
                .collect::<Vec<_>>()
        );

        // Reset brain weights về 0 trước khi merge
        // Lưu ý: weights nằm ở cấp Network, không phải trong TupleConfig
        for w_table in brain.weights.iter_mut() {
            for val in w_table.iter_mut() {
                *val = 0.0;
            }
        }

        // Softmax Weighted Merge
        for (idx, (_, trained_brain, _)) in thread_results.iter().enumerate() {
            let weight = softmax_weights[idx];

            // Duyệt qua từng bảng weights (Master Weights)
            for (table_idx, w_table) in trained_brain.weights.iter().enumerate() {
                // Cộng dồn vào bảng tương ứng của brain chính
                for (w_idx, val) in w_table.iter().enumerate() {
                    brain.weights[table_idx][w_idx] += val * weight;
                }
            }
        }

        let train_avg = best_score;
        println!(
            "   -> Best Config: E={:.3} S={:.3} M={:.3} D={:.3} (Avg: {:.0})",
            best_config.w_empty,
            best_config.w_snake,
            best_config.w_merge,
            best_config.w_disorder,
            train_avg
        );

        // ============================================================
        // BƯỚC 2: SELECT CONFIG PHASE (Đã tích hợp ở trên)
        // best_config đã được tìm ra từ Thread Results
        // ============================================================

        // In ra config thực tế sẽ dùng cho Eval (đã tính đến Hot Reload nếu có)
        let current_hot_val = *hot_config.read().unwrap();
        let mut actual_eval_config = best_config;
        if let Some(v) = current_hot_val.w_empty_override {
            actual_eval_config.w_empty = v;
        }
        if let Some(v) = current_hot_val.w_snake_override {
            actual_eval_config.w_snake = v;
        }
        if let Some(v) = current_hot_val.w_merge_override {
            actual_eval_config.w_merge = v;
        }
        if let Some(v) = current_hot_val.w_disorder_override {
            actual_eval_config.w_disorder = v;
        }

        println!(
            "📊 Evaluation Training ({} games) with Best Config...",
            eval_games
        );
        println!(
            "   Cfg Thực Tế: Empty={:.1} Snake={:.1} Merge={:.1} Disorder={:.1}",
            actual_eval_config.w_empty,
            actual_eval_config.w_snake,
            actual_eval_config.w_merge,
            actual_eval_config.w_disorder
        );

        // Clone model ổn định để train evaluation
        // let eval_brain = best_stable_brain.clone();

        // Best config đã in ở trên

        // Clone MERGED BRAIN để train evaluation
        // Đây là điểm quan trọng: Tận dụng kiến thức đã học và merge từ 100k games trước
        let mut eval_brain = brain.clone();

        // Gán config tối ưu vào eval_brain
        eval_brain.w_empty = best_config.w_empty;
        eval_brain.w_snake = best_config.w_snake;
        eval_brain.w_merge = best_config.w_merge;
        eval_brain.w_disorder = best_config.w_disorder;

        // Train thật 80k games với config mới trên nền merged brain
        // Truyền hot_config để có thể override bất cứ lúc nào!
        let (eval_avg, eval_max, trained_eval_brain, best_replay_opt) = run_evaluation_training(
            eval_brain,
            best_config,
            eval_games,
            num_threads,
            gamma,
            total_target_episodes,
            best_stable_brain.total_episodes + chunk_episodes, // Offset đã tăng lên
            training_policy,
            hot_config.clone(), // 🔥 TRUYỀN HOT CONFIG VÀO!
        );

        // Lưu replay tốt nhất nếu có
        if let Some(replay) = best_replay_opt {
            let replay_json = serde_json::to_string(&replay).unwrap();
            if let Err(e) = std::fs::write("best_replay.json", replay_json) {
                eprintln!("⚠️ Failed to save replay: {}", e);
            } else {
                println!(
                    "🎬 Saved Best Replay of this iteration (Score: {:.0}) to best_replay.json",
                    replay.score
                );
            }
        }

        let duration = loop_start.elapsed();
        println!(
            "   -> 📊 Eval Result: Avg = {:.2} (Max: {:.0}) | Record: {:.2}",
            eval_avg, eval_max, best_eval_avg
        );

        // ============================================================
        // BƯỚC 4: SO SÁNH & SAVE
        // Chỉ save nếu điểm Eval cao hơn kỷ lục cũ
        // ============================================================
        if eval_avg > best_eval_avg {
            println!("✅ NEW RECORD! ({:.2} > {:.2})", eval_avg, best_eval_avg);

            // Cập nhật kỷ lục
            best_eval_avg = eval_avg;

            // Cập nhật thông số vào Brain đã train để lưu
            // TÍNH TOÀN BỘ: 100k học bẩn + 50k học thật đều được ghi nhận
            let mut save_brain = trained_eval_brain;
            save_brain.total_episodes =
                best_stable_brain.total_episodes + chunk_episodes + eval_games;
            save_brain.best_overall_avg = eval_avg;

            // Config đã được gán trước khi train nên không cần gán lại

            // Save Checkpoint
            best_stable_brain = save_brain.clone(); // Cập nhật mốc neo
            let filename = format!("brain_ep_{}.msgpack", save_brain.total_episodes);
            if let Err(e) = save_brain.export_to_msgpack(&filename) {
                eprintln!("❌ Save Error: {}", e);
            } else {
                println!("💾 Saved checkpoint: {}", filename);
            }
        } else {
            println!(
                "❌ REJECTED. (Eval {:.2} <= Record {:.2})",
                eval_avg, best_eval_avg
            );
            println!("🔄 Discarding changes. Reverting to previous best.");
            // Không làm gì cả, vòng lặp sau sẽ tự clone lại từ best_stable_brain cũ
        }

        println!(
            "⏱️ Loop Time: {:.1}s\n-----------------------------------",
            duration.as_secs_f64()
        );
    }
}

fn find_latest_checkpoint() -> Option<u32> {
    let mut max_ep = 0;
    let mut found = false;

    if let Ok(entries) = fs::read_dir(".") {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                // Kiểm tra xem có đúng định dạng file không
                if name.starts_with("brain_ep_") && name.ends_with(".msgpack") {
                    let num_part = name
                        .trim_start_matches("brain_ep_")
                        .trim_end_matches(".msgpack");

                    if let Ok(ep) = num_part.parse::<u32>() {
                        println!("  🔍 Found: {} (Ep: {})", name, ep); // Log để bác thấy nó tìm được gì
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
        println!("✅ Auto-discovered latest checkpoint: Ep {}", max_ep);
        Some(max_ep)
    } else {
        println!("⚠️ No checkpoints found in current directory.");
        None
    }
}

// ... (Các hàm khác giữ nguyên: start_config_watcher, run_training_parallel) ...
// Nhớ copy nốt hàm run_training_parallel ở code trước vào nhé!
fn start_config_watcher(shared_hot_config: Arc<RwLock<HotLoadConfig>>) {
    thread::spawn(move || {
        let mut last_cfg = HotLoadConfig::default();
        loop {
            thread::sleep(Duration::from_secs(2));
            if let Ok(file) = File::open("config.json") {
                let reader = BufReader::new(file);
                match serde_json::from_reader::<_, HotLoadConfig>(reader) {
                    Ok(new_cfg) => {
                        let mut changed = false;
                        // Kiểm tra thay đổi weights
                        if new_cfg.w_empty_override != last_cfg.w_empty_override
                            || new_cfg.w_snake_override != last_cfg.w_snake_override
                            || new_cfg.w_merge_override != last_cfg.w_merge_override
                            || new_cfg.w_disorder_override != last_cfg.w_disorder_override
                        {
                            changed = true;
                        }

                        // ✅ SỬA: So sánh đúng tên biến alpha_override
                        if new_cfg.alpha_override != last_cfg.alpha_override {
                            changed = true;
                        }
                        // ✅ SỬA: Thêm check epsilon thay đổi
                        if new_cfg.epsilon_override != last_cfg.epsilon_override {
                            changed = true;
                        }

                        if changed {
                            print!("\n🔥 HOT RELOAD (Overrides): ");
                            if let Some(v) = new_cfg.w_empty_override {
                                print!("Empty={:.1} ", v);
                            }
                            if let Some(v) = new_cfg.w_snake_override {
                                print!("Snake={:.1} ", v);
                            }
                            if let Some(v) = new_cfg.w_merge_override {
                                print!("Merge={:.1} ", v);
                            }
                            if let Some(v) = new_cfg.w_disorder_override {
                                print!("DisOrder={:.1} ", v);
                            }

                            // ✅ SỬA: In đúng tên biến
                            if let Some(v) = new_cfg.alpha_override {
                                print!("α={:.4} ", v);
                            }
                            if let Some(v) = new_cfg.epsilon_override {
                                print!("ε={:.4} ", v);
                            }
                            if let Some(v) = new_cfg.eval_epsilon_override {
                                print!("Ev_ε={:.4} ", v);
                            }

                            println!();
                            last_cfg = new_cfg.clone();
                        }
                        let mut write_guard = shared_hot_config.write().unwrap();
                        *write_guard = new_cfg;
                    }
                    Err(_e) => {}
                }
            }
        }
    });
}

/// Hàm Evaluation Training: TRAIN THẬT với config cố định
/// Khác với run_training_parallel:
/// - Không dùng PBT evolve (config cố định)
/// - Không có Hot Reload
/// - Trả về brain đã train để có thể save
fn run_evaluation_training(
    mut brain: NTupleNetwork, // Ownership - sẽ được train và trả về
    config: TrainingConfig,   // Config cơ sở để train
    total_games: u32,         // Số lượng game (vd: 80,000)
    num_threads: u32,
    gamma: f64,
    total_target_episodes: u32,
    start_offset: u32,
    policy: TrainingPolicy,
    hot_config: Arc<RwLock<HotLoadConfig>>, // 🔥 HOT CONFIG ĐÃ THÊM!
) -> (f64, f64, NTupleNetwork, Option<GameReplay>) {
    // Trả về (Avg Score, Max Score, Trained Brain, Best Replay)

    // Tạo shared pointer để các thread cùng update weights
    let brain_ptr = SharedBrain {
        network: &mut brain as *mut NTupleNetwork,
    };
    let shared_brain = Arc::new(brain_ptr);

    let ep_per_thread = total_games / num_threads;

    // Chạy song song - mỗi thread train một phần games
    // Return: (local_scores, local_best_replay)
    let results: Vec<(Vec<f64>, Option<GameReplay>)> = (0..num_threads)
        .into_par_iter()
        .map(|t_id| {
            let mut local_env = ThreesEnv::new(gamma);

            let ptr = shared_brain.network;
            let local_brain = unsafe { &mut *ptr };

            let mut local_scores = Vec::with_capacity(ep_per_thread as usize);
            let mut rng = rand::rng();

            let mut local_best_replay: Option<GameReplay> = None;
            let mut local_max_score = 0.0;

            for local_ep in 0..ep_per_thread {
                let current_global_ep = (local_ep * num_threads + t_id) + start_offset;
                let progress = current_global_ep as f64 / total_target_episodes as f64;

                // ============ HOT RELOAD - ĐỌC CONFIG.JSON ============
                let current_hot = *hot_config.read().unwrap();
                let mut effective_config = config; // Bắt đầu từ config cơ sở

                if let Some(v) = current_hot.w_empty_override {
                    effective_config.w_empty = v;
                }
                if let Some(v) = current_hot.w_snake_override {
                    effective_config.w_snake = v;
                }
                if let Some(v) = current_hot.w_merge_override {
                    effective_config.w_merge = v;
                }
                if let Some(v) = current_hot.w_disorder_override {
                    effective_config.w_disorder = v;
                }

                local_env.set_config(effective_config);

                // Cập nhật thông số vào brain để khi Save nó mang thông số mới này!
                local_brain.w_empty = effective_config.w_empty;
                local_brain.w_snake = effective_config.w_snake;
                local_brain.w_merge = effective_config.w_merge;
                local_brain.w_disorder = effective_config.w_disorder;
                // ========================================================

                // Alpha & Epsilon decay
                let mut current_alpha = (0.01 * (1.0 - progress)).max(0.0001);
                if let Some(v) = current_hot.alpha_override {
                    current_alpha = v;
                }
                // 🔥 Dùng eval_epsilon từ config nếu có, không thì dùng epsilon decay
                let current_epsilon = current_hot
                    .eval_epsilon_override
                    .unwrap_or_else(|| (0.2 * (1.0 - (progress / 0.8))).max(0.01));

                // GAME LOOP
                local_env.reset();
                // local_brain.reset_traces(); // Removed: Traces now managed by env

                // TRACKING FOR REPLAY
                let mut current_steps = Vec::new();
                let initial_board_state = {
                    let mut arr = [[0u32; 4]; 4];
                    for r in 0..4 {
                        for c in 0..4 {
                            arr[r][c] = local_env.game.board[r][c].value;
                        }
                    }
                    arr
                };

                let mut step_count = 0;
                while !local_env.game.game_over {
                    step_count += 1;
                    if step_count > 20000 {
                        break;
                    }

                    let action = if rng.random_bool(current_epsilon.into()) {
                        local_env.get_random_valid_action()
                    } else {
                        local_env.get_best_action_expectimax(local_brain)
                    };

                    // Execute Step
                    local_env.train_step(local_brain, action, current_alpha);

                    // Record Step
                    let board_snap = {
                        let mut arr = [[0u32; 4]; 4];
                        for r in 0..4 {
                            for c in 0..4 {
                                arr[r][c] = local_env.game.board[r][c].value;
                            }
                        }
                        arr
                    };

                    current_steps.push(StepData {
                        direction: action as usize,
                        board: board_snap,
                        score: local_env.game.score,
                    });
                }

                let game_score = local_env.game.score as f64;
                local_scores.push(game_score);

                // Update Request Best Replay
                if game_score > local_max_score {
                    local_max_score = game_score;
                    local_best_replay = Some(GameReplay {
                        score: game_score,
                        max_tile: local_env.game.get_highest_tile_value(),
                        initial_board: initial_board_state,
                        steps: current_steps,
                    });
                }

                // Log progress (chỉ thread 0)
                if t_id == 0 && local_ep % 1000 == 0 {
                    print!(
                        "\r   Eval: {:>6}/{} | Last: {:>5.0} | HotCfg: S{:.0} M{:.0}   ",
                        local_ep * num_threads,
                        total_games,
                        local_env.game.score,
                        effective_config.w_snake,
                        effective_config.w_merge
                    );
                    use std::io::Write;
                    std::io::stdout().flush().unwrap();
                }
            }

            (local_scores, local_best_replay)
        })
        .collect();

    println!(); // Newline sau progress

    let all_scores: Vec<f64> = results.iter().map(|(s, _)| s).flatten().cloned().collect();
    let avg = all_scores.iter().sum::<f64>() / all_scores.len() as f64;
    let max = all_scores.iter().fold(0.0f64, |a, &b| a.max(b));

    // Find global best replay
    let mut global_best_replay: Option<GameReplay> = None;
    let mut global_max_score = 0.0;

    for (_, replay_opt) in results {
        if let Some(replay) = replay_opt {
            if replay.score > global_max_score {
                global_max_score = replay.score;
                global_best_replay = Some(replay);
            }
        }
    }

    // Tính điểm trung bình của 10% thấp nhất (Bottom 10%)
    let mut sorted_scores = all_scores.clone();
    sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let bot10_count = (sorted_scores.len() as f64 * 0.1).ceil() as usize;
    let bot10_avg: f64 = sorted_scores.iter().take(bot10_count).sum::<f64>() / bot10_count as f64;

    println!(
        "   📉 Bottom 10% Avg: {:.2} ({} games)",
        bot10_avg, bot10_count
    );

    (avg, max, brain, global_best_replay)
}
