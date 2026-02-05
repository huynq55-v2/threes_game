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

    println!("🚀 Bắt đầu Training với Logic: Top 1% Average & Strict Auto-Revert...");
    println!(
        "📊 Current Record: Top1% Avg = {:.2} (tại Ep {})",
        best_stable_brain.best_top1_avg, best_stable_brain.total_episodes
    );

    loop {
        let start_time = std::time::Instant::now();

        // Bước 0: LUÔN RESET VỀ TRẠNG THÁI ỔN ĐỊNH NHẤT
        // Brain nháp (mutable) được tạo ra từ bản chuẩn.
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
        let buff_idx = rng.random_range(0..4);

        match buff_idx {
            0 => {
                brain.w_empty *= buff_multiplier;
                print!("✨ BUFF EMPTY! ");
            }
            1 => {
                brain.w_snake *= buff_multiplier;
                print!("🐍 BUFF SNAKE! ");
            }
            2 => {
                brain.w_merge *= buff_multiplier;
                print!("🔗 BUFF MERGE! ");
            }
            _ => {
                brain.w_disorder *= buff_multiplier;
                print!("⚡ BUFF DISORDER! ");
            }
        }

        println!(
            "-> Test Config: {:.1}/{:.1}/{:.1}/{:.1}",
            brain.w_empty, brain.w_snake, brain.w_merge, brain.w_disorder
        );

        // if 1 of 4 params larger than 10000 then buff_multiplier = 1.0 / GOLDEN_RATIO
        if brain.w_empty > 100000.0
            || brain.w_snake > 100000.0
            || brain.w_merge > 100000.0
            || brain.w_disorder > 100000.0
        {
            buff_multiplier = 1.0 / GOLDEN_RATIO;
        }

        // if 1 of 4 params smaller than 50 then buff_multiplier = GOLDEN_RATIO
        if brain.w_empty < 60.0
            || brain.w_snake < 60.0
            || brain.w_merge < 60.0
            || brain.w_disorder < 60.0
        {
            buff_multiplier = GOLDEN_RATIO;
        }

        println!("-> Buff Multiplier: {:.2}", buff_multiplier);

        // ------------------------------------------------------
        // 2. CHẠY SONG SONG
        // ------------------------------------------------------
        let ep_per_thread = chunk_episodes as u32 / num_threads;

        // Lấy mốc thời gian hiện tại để tính Alpha/Epsilon
        let current_base_ep = best_stable_brain.total_episodes;
        // Mục tiêu của vòng này là chạy thêm chunk_episodes
        let target_ep = current_base_ep + chunk_episodes;

        let results: Vec<Vec<f64>> = (0..num_threads)
            .into_par_iter()
            .map(|t_id| {
                let mut local_env = ThreesEnv::new(gamma);

                run_training_parallel(
                    &mut local_env,
                    shared_brain_loop.clone(),
                    pbt_manager.clone(),
                    hot_config.clone(),
                    ep_per_thread,
                    total_target_episodes,
                    current_base_ep, // Start offset
                    t_id,
                    num_threads,
                    training_policy,
                    buff_multiplier,
                )
            })
            .collect();

        let mut all_scores: Vec<f64> = results.into_iter().flatten().collect();

        // ------------------------------------------------------
        // 3. TÍNH TOÁN METRIC (3 TIÊU CHÍ)
        // ------------------------------------------------------
        all_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let total_count = all_scores.len();

        // A. Top 1%
        let top_1_count = (total_count as f64 * 0.01).ceil() as usize;
        let top_1_count = top_1_count.max(1);
        let top_1_avg: f64 = all_scores[0..top_1_count].iter().sum::<f64>() / top_1_count as f64;

        // B. Average
        let overall_avg: f64 = all_scores.iter().sum::<f64>() / total_count as f64;

        // C. Bottom 10%
        let bot_10_count = (total_count as f64 * 0.1).ceil() as usize;
        let bot_10_count = bot_10_count.max(1);
        let bot_10_avg: f64 =
            all_scores[total_count - bot_10_count..].iter().sum::<f64>() / bot_10_count as f64;

        let duration = start_time.elapsed();
        println!("\n📊 Stats Loop (Target Ep {}):", target_ep);
        println!(
            "   - Top 1% Avg:   {:.2} (Rec: {:.2})",
            top_1_avg, best_stable_brain.best_top1_avg
        );
        println!(
            "   - Overall Avg:  {:.2} (Rec: {:.2})",
            overall_avg, best_stable_brain.best_overall_avg
        );
        println!(
            "   - Bot 10% Avg:  {:.2} (Rec: {:.2})",
            bot_10_avg, best_stable_brain.best_bot10_avg
        );

        // ------------------------------------------------------
        // 4. QUYẾT ĐỊNH
        // ------------------------------------------------------

        // Điều kiện: Tốt hơn ở CẢ 3 chỉ số
        // Mẹo: Dùng >= cho 2 chỉ số phụ để dễ thở hơn chút, > cho chỉ số chính
        let is_better = top_1_avg > best_stable_brain.best_top1_avg
            && overall_avg >= best_stable_brain.best_overall_avg
            && bot_10_avg >= best_stable_brain.best_bot10_avg;

        if is_better {
            println!("✅ NEW RECORD! Thỏa mãn 3 tiêu chí.");

            // 1. Cập nhật Stats vào Brain
            brain.total_episodes = target_ep; // CHỐT SỐ EPISODE MỚI TẠI ĐÂY
            brain.best_top1_avg = top_1_avg;
            brain.best_overall_avg = overall_avg;
            brain.best_bot10_avg = bot_10_avg;

            // 2. Cập nhật Config PBT
            {
                let pbt = pbt_manager.lock().unwrap();
                if let Some(best_thread) = pbt.get_best_config_entry() {
                    let best_cfg = best_thread.1;
                    brain.w_empty = best_cfg.w_empty;
                    brain.w_snake = best_cfg.w_snake;
                    brain.w_merge = best_cfg.w_merge;
                    brain.w_disorder = best_cfg.w_disorder;
                }
            }

            // 3. LƯU CHECKPOINT CỨNG
            // Lần sau loop sẽ clone từ bản này
            best_stable_brain = brain.clone();

            // 4. Lưu File
            // Tên file lấy trực tiếp từ brain.total_episodes -> KHÔNG BAO GIỜ SAI ĐƯỢC
            let filename = format!("brain_ep_{}.msgpack", brain.total_episodes);
            if let Err(e) = brain.export_to_msgpack(&filename) {
                eprintln!("❌ Lỗi lưu file: {}", e);
            } else {
                println!("💾 Saved checkpoint: {}", filename);
            }
        } else {
            println!("❌ FAILED. Không đủ chuẩn.");
            println!(
                "   (Yêu cầu: Top1>{:.2}, Avg>={:.2}, Bot10>={:.2})",
                best_stable_brain.best_top1_avg,
                best_stable_brain.best_overall_avg,
                best_stable_brain.best_bot10_avg
            );

            println!("🔄 Reverting... Về Ep {}", best_stable_brain.total_episodes);
            // KHÔNG LÀM GÌ CẢ. Brain tự reset ở đầu vòng lặp.
        }

        println!(
            "⏱️ Time: {:.1}s\n-----------------------------------------------------------",
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

fn run_verification_parallel(
    brain: &NTupleNetwork,  // Truyền tham chiếu (Read-only)
    config: TrainingConfig, // Config muốn test
    total_games: u32,       // Số lượng game (vd: 50,000)
    num_threads: u32,
) -> (f64, f64) {
    // Trả về (Avg Score, Max Score)

    // Chia việc cho các luồng
    let scores: Vec<f64> = (0..total_games)
        .into_par_iter() // Rayon parallel iterator
        .map(|_| {
            // Mỗi game tạo một môi trường mới sạch sẽ
            let mut env = ThreesEnv::new(0.0); // Gamma không quan trọng khi test
            env.set_config(config);

            // Clone não để dùng (chỉ đọc weight, không ghi)
            // Lưu ý: NTupleNetwork của bạn phải derive Clone
            let mut local_brain = brain.clone();

            env.reset();
            let mut step_count = 0;

            while !env.game.game_over && step_count < 20000 {
                step_count += 1;

                // CHƠI NGHIÊM TÚC: Expectimax (hoặc Greedy tùy bạn chọn)
                // Tuyệt đối không có Random Move ở đây (trừ khi tiles ra ngẫu nhiên)
                let action = env.get_best_action_expectimax(&mut local_brain);

                // Chỉ đi nước bước, KHÔNG TRAIN
                env.game.step(action);
            }

            env.game.score as f64
        })
        .collect();

    let avg = scores.iter().sum::<f64>() / total_games as f64;
    let max = scores.iter().fold(0.0f64, |a, &b| a.max(b));

    (avg, max)
}
