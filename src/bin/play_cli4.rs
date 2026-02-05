use rand::Rng;
use rayon::prelude::*;
use std::fs::{self, File}; // Thêm fs để quét thư mục
use std::io::BufReader;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use std::{env, thread};
use threes_rs::hotload_config::HotLoadConfig;
use threes_rs::{
    n_tuple_network::NTupleNetwork, pbt::PBTManager, pbt::TrainingConfig, python_module::ThreesEnv,
};

// Hằng số Tỷ lệ vàng
const GOLDEN_RATIO: f64 = 1.61803398875;

// Struct wrapper pointer (giữ nguyên)
struct SharedBrain {
    network: *mut NTupleNetwork,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum TrainingPolicy {
    Greedy,
    Expectimax,
}

unsafe impl Send for SharedBrain {}
unsafe impl Sync for SharedBrain {}

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
        "greedy" => {
            println!("⚡ Training Mode: GREEDY");
            TrainingPolicy::Greedy
        }
        _ => {
            println!("🧠 Training Mode: EXPECTIMAX");
            TrainingPolicy::Expectimax
        }
    };

    let multiplier = args[2].to_lowercase();

    let mut buff_multiplier = 1.0;
    if multiplier == "mul" {
        buff_multiplier = GOLDEN_RATIO;
    } else if multiplier == "div" {
        buff_multiplier = 1.0 / GOLDEN_RATIO;
    }

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
    if brain.w_empty == 0.0 {
        brain.w_empty = 50.0;
    }
    if brain.w_snake == 0.0 {
        brain.w_snake = 50.0;
    }
    if brain.w_merge == 0.0 {
        brain.w_merge = 50.0;
    }
    if brain.w_disorder == 0.0 {
        brain.w_disorder = 50.0;
    }

    // Config Watcher & PBT
    let hot_config = Arc::new(RwLock::new(HotLoadConfig::default()));
    start_config_watcher(hot_config.clone());
    let pbt_manager = Arc::new(Mutex::new(PBTManager::new()));

    let chunk_episodes = 100_000;
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
    let eval_games = 50_000u32;

    loop {
        let loop_start = std::time::Instant::now();

        // ============================================================
        // BƯỚC 0: RESET VỀ BẢN CHUẨN ĐỂ TRAIN
        // ============================================================
        brain = best_stable_brain.clone();

        // Tạo pointer MỚI cho vòng lặp này (Quan trọng!)
        let brain_ptr = SharedBrain {
            network: &mut brain as *mut NTupleNetwork,
        };
        let shared_brain_loop = Arc::new(brain_ptr);

        // ------------------------------------------------------
        // 1. LOGIC BUFF (Random 1 chỉ số)
        // ------------------------------------------------------
        let mut rng = rand::rng();
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

        // if 1 of 4 params larger than 10000 then buff_multiplier = 1.0 / GOLDEN_RATIO
        if brain.w_empty > 100000.0
            || brain.w_snake > 100000.0
            || brain.w_merge > 100000.0
            || brain.w_disorder > 100000.0 || brain.w_empty < 60.0
            || brain.w_snake < 60.0
            || brain.w_merge < 60.0
            || brain.w_disorder < 60.0
        {
            buff_multiplier = 1.0 / buff_multiplier;
        }

        println!("-> Buff Multiplier: {:.2}", buff_multiplier);

        // ============================================================
        // BƯỚC 1: PARALLEL TRAINING - Mỗi thread clone brain riêng
        // Mục đích: Tìm CONFIG tối ưu cho brain hiện tại
        // ============================================================
        println!("🏋️ Training Phase ({} games, {} threads)...", chunk_episodes, num_threads);

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
                    let mutate_factor = if rng.random_bool(0.5) { 
                        buff_multiplier 
                    } else { 
                        1.0
                    };
                    
                    match param_idx {
                        0 => thread_config.w_empty *= mutate_factor,
                        1 => thread_config.w_snake *= mutate_factor,
                        2 => thread_config.w_merge *= mutate_factor,
                        _ => thread_config.w_disorder *= mutate_factor,
                    }
                }

                local_env.set_config(thread_config);
                local_brain.w_empty = thread_config.w_empty;
                local_brain.w_snake = thread_config.w_snake;
                local_brain.w_merge = thread_config.w_merge;
                local_brain.w_disorder = thread_config.w_disorder;

                // Train riêng biệt
                let mut total_score = 0.0;
                for local_ep in 0..ep_per_thread {
                    let current_global_ep = (local_ep * num_threads + t_id) + current_base_ep;
                    let progress = current_global_ep as f64 / total_target_episodes as f64;

                    let current_alpha = (0.01 * (1.0 - progress)).max(0.0001);
                    let current_epsilon = (0.2 * (1.0 - (progress / 0.8))).max(0.01);

                    local_env.reset();
                    let mut step_count = 0;
                    while !local_env.game.game_over {
                        step_count += 1;
                        if step_count > 20000 { break; }

                        let action = if rng.random_bool(current_epsilon.into()) {
                            local_env.get_random_valid_action()
                        } else {
                            match training_policy {
                                TrainingPolicy::Greedy => local_env.get_best_action_greedy(&mut local_brain),
                                TrainingPolicy::Expectimax => local_env.get_best_action_expectimax(&mut local_brain),
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
                            local_ep, ep_per_thread, running_avg, 
                            thread_config.w_snake, thread_config.w_merge
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

        // Merge weights bằng Weighted Average (theo điểm)
        let total_weight: f64 = thread_results.iter().map(|(s, _, _)| s).sum();
        
        // Reset brain weights về 0 trước khi merge
        for tuple in brain.tuples.iter_mut() {
            for w in tuple.weights.iter_mut() {
                *w = 0.0;
            }
        }

        // Weighted average merge
        for (score, trained_brain, _) in thread_results.iter() {
            let weight = score / total_weight;
            for (i, tuple) in trained_brain.tuples.iter().enumerate() {
                for (j, w) in tuple.weights.iter().enumerate() {
                    brain.tuples[i].weights[j] += w * weight;
                }
            }
        }

        let train_avg = best_score;
        println!("   -> Best Config: E={:.0} S={:.0} M={:.0} D={:.0} (Avg: {:.0})",
            best_config.w_empty, best_config.w_snake, best_config.w_merge, best_config.w_disorder, train_avg
        );

        // ============================================================
        // BƯỚC 2: SELECT CONFIG PHASE (Đã tích hợp ở trên)
        // best_config đã được tìm ra từ Thread Results
        // ============================================================

        // ============================================================
        // BƯỚC 3: EVALUATION TRAINING (50k games)
        // Lấy MODEL ỔN ĐỊNH + CONFIG MỚI, TRAIN THẬT để đánh giá
        // ============================================================
        println!(
            "📊 Evaluation Training ({} games) with Best Config...",
            eval_games
        );
        println!(
            "   Cfg: Empty={:.1} Snake={:.1} Merge={:.1} Disorder={:.1}",
            best_config.w_empty,
            best_config.w_snake,
            best_config.w_merge,
            best_config.w_disorder
        );

        // Clone model ổn định để train evaluation
        let mut eval_brain = best_stable_brain.clone();
        
        // Gán config mới vào eval_brain
        eval_brain.w_empty = best_config.w_empty;
        eval_brain.w_snake = best_config.w_snake;
        eval_brain.w_merge = best_config.w_merge;
        eval_brain.w_disorder = best_config.w_disorder;

        // Train thật 50k games với config mới
        let (eval_avg, eval_max, trained_eval_brain) = run_evaluation_training(
            eval_brain,
            best_config,
            eval_games,
            num_threads,
            gamma,
            total_target_episodes,
            best_stable_brain.total_episodes,
            training_policy,
        );

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
            println!(
                "✅ NEW RECORD! ({:.2} > {:.2})",
                eval_avg, best_eval_avg
            );

            // Cập nhật kỷ lục
            best_eval_avg = eval_avg;

            // Cập nhật thông số vào Brain đã train để lưu
            // Chỉ tính eval_games vì 100k train noisy chỉ dùng để tìm config
            let mut save_brain = trained_eval_brain;
            save_brain.total_episodes = best_stable_brain.total_episodes + eval_games;
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
    thread::spawn(move || loop {
        thread::sleep(Duration::from_secs(2));
        if let Ok(file) = File::open("config.json") {
            let reader = BufReader::new(file);
            if let Ok(new_cfg) = serde_json::from_reader(reader) {
                let mut write_guard = shared_hot_config.write().unwrap();
                *write_guard = new_cfg;
            }
        }
    });
}

fn run_training_parallel(
    env: &mut ThreesEnv,
    shared_brain: Arc<SharedBrain>,
    pbt: Arc<Mutex<PBTManager>>,
    hot_config: Arc<RwLock<HotLoadConfig>>,
    episodes_to_run: u32,
    total_target_episodes: u32,
    start_offset: u32,
    thread_id: u32,
    num_threads: u32,
    policy: TrainingPolicy,
    buff_multiplier: f64,
) -> Vec<f64> {
    // <--- Thay đổi kiểu trả về
    let mut rng = rand::rng();
    let mut running_error = 0.0;
    let mut running_score = 0.0;

    // Vector lưu điểm số của thread này
    let mut local_scores = Vec::with_capacity(episodes_to_run as usize);

    // LẤY NÃO (UNSAFE)
    let ptr = shared_brain.network;
    let brain = unsafe { &mut *ptr };

    // KHỞI TẠO CONFIG CHO THREAD NÀY
    let mut pbt_config = {
        TrainingConfig {
            w_empty: brain.w_empty,
            w_snake: brain.w_snake,
            w_merge: brain.w_merge,
            w_disorder: brain.w_disorder,
        }
    };

    for local_ep in 0..episodes_to_run {
        let current_global_ep = (local_ep * num_threads + thread_id) + start_offset;
        let progress = current_global_ep as f64 / total_target_episodes as f64;

        // HOT RELOAD
        let current_hot = *hot_config.read().unwrap();
        let mut effective_config = pbt_config;
        if current_hot.w_empty_override > 0.0 {
            effective_config.w_empty = current_hot.w_empty_override;
        }
        if current_hot.w_snake_override > 0.0 {
            effective_config.w_snake = current_hot.w_snake_override;
        }
        if current_hot.w_merge_override > 0.0 {
            effective_config.w_merge = current_hot.w_merge_override;
        }
        if current_hot.w_disorder_override > 0.0 {
            effective_config.w_disorder = current_hot.w_disorder_override;
        }

        env.set_config(effective_config);

        // Alpha & Epsilon
        let mut current_alpha = (0.01 * (1.0 - progress)).max(0.0001);
        if current_hot.alpha_override > 0.0 {
            current_alpha = current_hot.alpha_override;
        }
        let current_epsilon = (0.2 * (1.0 - (progress / 0.8))).max(0.01);

        // GAME LOOP
        env.reset();
        let mut step_count = 0;
        while !env.game.game_over {
            step_count += 1;
            if step_count > 20000 {
                break;
            }

            let action = if rng.random_bool(current_epsilon.into()) {
                env.get_random_valid_action()
            } else {
                match policy {
                    TrainingPolicy::Greedy => env.get_best_action_greedy(brain),
                    TrainingPolicy::Expectimax => env.get_best_action_expectimax(brain),
                }
            };

            let (error, _) = env.train_step(brain, action, current_alpha);
            running_error = running_error * 0.999 + error * 0.001;
        }

        let final_score = env.game.score as f64;
        running_score = running_score * 0.99 + final_score * 0.01;

        // Push điểm vào list
        local_scores.push(final_score);

        // PBT EVOLVE
        if local_ep > 0 && local_ep % 1000 == 0 {
            let mut pbt_guard = pbt.lock().unwrap();
            let (evolved, new_cfg) =
                pbt_guard.report_and_evolve(thread_id, running_score, pbt_config, buff_multiplier);
            if evolved {
                pbt_config = new_cfg;
            }
        }

        if thread_id == 0 && local_ep % 1000 == 0 {
            print!(
                "\r   Run: {:>6} | Sc(EMA): {:>5.0} | Cfg: S{:.0} M{:.0}   ",
                local_ep, running_score, effective_config.w_snake, effective_config.w_merge
            );
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
    }
    if thread_id == 0 {
        println!();
    }

    // Trả về danh sách điểm
    local_scores
}

/// Hàm Evaluation Training: TRAIN THẬT với config cố định
/// Khác với run_training_parallel:
/// - Không dùng PBT evolve (config cố định)
/// - Không có Hot Reload
/// - Trả về brain đã train để có thể save
fn run_evaluation_training(
    mut brain: NTupleNetwork,    // Ownership - sẽ được train và trả về
    config: TrainingConfig,      // Config cố định để train
    total_games: u32,            // Số lượng game (vd: 50,000)
    num_threads: u32,
    gamma: f64,
    total_target_episodes: u32,
    start_offset: u32,
    policy: TrainingPolicy,
) -> (f64, f64, NTupleNetwork) {
    // Trả về (Avg Score, Max Score, Trained Brain)

    // Tạo shared pointer để các thread cùng update weights
    let brain_ptr = SharedBrain {
        network: &mut brain as *mut NTupleNetwork,
    };
    let shared_brain = Arc::new(brain_ptr);

    let ep_per_thread = total_games / num_threads;

    // Chạy song song - mỗi thread train một phần games
    let results: Vec<Vec<f64>> = (0..num_threads)
        .into_par_iter()
        .map(|t_id| {
            let mut local_env = ThreesEnv::new(gamma);
            local_env.set_config(config); // Config cố định

            let ptr = shared_brain.network;
            let local_brain = unsafe { &mut *ptr };

            let mut local_scores = Vec::with_capacity(ep_per_thread as usize);
            let mut rng = rand::rng();

            for local_ep in 0..ep_per_thread {
                let current_global_ep = (local_ep * num_threads + t_id) + start_offset;
                let progress = current_global_ep as f64 / total_target_episodes as f64;

                // Alpha & Epsilon decay
                let current_alpha = (0.01 * (1.0 - progress)).max(0.0001);
                let current_epsilon = (0.2 * (1.0 - (progress / 0.8))).max(0.01);

                // GAME LOOP
                local_env.reset();
                let mut step_count = 0;
                while !local_env.game.game_over {
                    step_count += 1;
                    if step_count > 20000 {
                        break;
                    }

                    let action = if rng.random_bool(current_epsilon.into()) {
                        local_env.get_random_valid_action()
                    } else {
                        match policy {
                            TrainingPolicy::Greedy => local_env.get_best_action_greedy(local_brain),
                            TrainingPolicy::Expectimax => {
                                local_env.get_best_action_expectimax(local_brain)
                            }
                        }
                    };

                    // TRAIN THẬT - update weights
                    local_env.train_step(local_brain, action, current_alpha);
                }

                local_scores.push(local_env.game.score as f64);

                // Log progress (chỉ thread 0)
                if t_id == 0 && local_ep % 1000 == 0 {
                    print!(
                        "\r   Eval: {:>6}/{} | Last: {:>5.0}   ",
                        local_ep * num_threads,
                        total_games,
                        local_env.game.score
                    );
                    use std::io::Write;
                    std::io::stdout().flush().unwrap();
                }
            }

            local_scores
        })
        .collect();

    println!(); // Newline sau progress

    let all_scores: Vec<f64> = results.into_iter().flatten().collect();
    let avg = all_scores.iter().sum::<f64>() / all_scores.len() as f64;
    let max = all_scores.iter().fold(0.0f64, |a, &b| a.max(b));

    (avg, max, brain)
}

