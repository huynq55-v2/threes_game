use rand::Rng;
use rayon::prelude::*;
use std::fs::File;
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

    // Episode override (nếu muốn force từ dòng lệnh, không thì lấy từ file)
    let override_episode = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(0) as u32
    } else {
        0 as u32
    };

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

    let multiplier = args[3].to_lowercase();

    let mut BUFF_MULTIPLIER = 1.0;

    if multiplier == "mul" {
        BUFF_MULTIPLIER = GOLDEN_RATIO;
    } else if multiplier == "div" {
        BUFF_MULTIPLIER = 1.0 / GOLDEN_RATIO;
    }

    println!("Multiplier: {}", multiplier);

    // --- SETUP BRAIN ---
    let mut brain = if override_episode > 0 {
        let filename = format!("brain_ep_{}.msgpack", override_episode);
        println!("📂 Loading brain: {}", filename);
        let b = NTupleNetwork::load_from_msgpack(&filename).expect("Không tìm thấy file!");
        println!(
            "🧐 LOAD DATA: E={:.1}, S={:.1}, M={:.1}, D={:.1}",
            b.w_empty, b.w_snake, b.w_merge, b.w_disorder
        );
        b
    } else {
        println!("✨ Tạo não mới tinh...");
        NTupleNetwork::new(0.1, gamma)
    };

    if override_episode > 0 && brain.total_episodes == 0 {
        brain.total_episodes = override_episode;
    }

    let mut current_global_episode = brain.total_episodes;

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

    // Pointer setup
    let brain_ptr = SharedBrain {
        network: &mut brain as *mut NTupleNetwork,
    };
    let shared_brain = Arc::new(brain_ptr);

    // Config Watcher & PBT
    let hot_config = Arc::new(RwLock::new(HotLoadConfig::default()));
    start_config_watcher(hot_config.clone());
    let pbt_manager = Arc::new(Mutex::new(PBTManager::new()));

    let chunk_episodes = 100_000;
    let total_target_episodes = 100_000_000;

    // --- LOGIC MỚI: THEO DÕI BEST TOP 1% ---
    // Khởi tạo mức điểm chuẩn ban đầu (có thể set 0 hoặc chạy thử 1 vòng test để lấy)
    let mut best_top1_percent_avg = 0.0;

    // Backup não tốt nhất hiện tại (Deep Clone)
    // Lưu ý: NTupleNetwork phải hỗ trợ Clone. Nếu chưa có, bạn cần thêm #[derive(Clone)] vào struct NTupleNetwork
    let mut best_stable_brain = brain.clone();

    println!("🚀 Bắt đầu Training với Logic: Top 1% Average & Auto-Revert...");

    loop {
        let start_time = std::time::Instant::now();

        // Bước 0: LUÔN RESET VỀ TRẠNG THÁI ỔN ĐỊNH NHẤT
        // Nếu vòng trước fail, brain sẽ quay về như chưa từng train vòng đó.
        // Nếu vòng trước win, best_stable_brain đã được update ở cuối vòng lặp.
        brain = best_stable_brain.clone();

        // Cập nhật lại pointer trỏ vào brain mới reset
        // (Lưu ý: SharedBrain dùng pointer raw, nên ta gán đè dữ liệu vào vùng nhớ cũ
        // hoặc tạo Arc mới. Ở đây vì cấu trúc main, ta cần đảm bảo pointer trỏ đúng)
        // Cách an toàn nhất trong loop này là tạo SharedBrain mới cho mỗi vòng lặp
        // vì biến `brain` bị move/clone lại.
        let brain_ptr = SharedBrain {
            network: &mut brain as *mut NTupleNetwork,
        };
        let shared_brain_loop = Arc::new(brain_ptr);

        // ------------------------------------------------------
        // 1. LOGIC BUFF (Random 1 chỉ số)
        // ------------------------------------------------------
        let mut rng = rand::rng();
        let buff_idx = rng.random_range(0..4);
        let old_vals = (
            brain.w_empty,
            brain.w_snake,
            brain.w_merge,
            brain.w_disorder,
        );

        match buff_idx {
            0 => {
                brain.w_empty *= BUFF_MULTIPLIER;
                print!("✨ BUFF EMPTY! ");
            }
            1 => {
                brain.w_snake *= BUFF_MULTIPLIER;
                print!("🐍 BUFF SNAKE! ");
            }
            2 => {
                brain.w_merge *= BUFF_MULTIPLIER;
                print!("🔗 BUFF MERGE! ");
            }
            _ => {
                brain.w_disorder *= BUFF_MULTIPLIER;
                print!("⚡ BUFF DISORDER! ");
            }
        }

        println!(
            "-> Test Config: {:.1}/{:.1}/{:.1}/{:.1}",
            brain.w_empty, brain.w_snake, brain.w_merge, brain.w_disorder
        );

        // ------------------------------------------------------
        // 2. CHẠY SONG SONG & THU THẬP ĐIỂM SỐ
        // ------------------------------------------------------
        let ep_per_thread = chunk_episodes as u32 / num_threads;

        let current_ep_for_decay = brain.total_episodes;

        // Sử dụng map của rayon để thu về vector điểm số từ các luồng
        let results: Vec<Vec<f64>> = (0..num_threads)
            .into_par_iter()
            .map(|t_id| {
                let mut local_env = ThreesEnv::new(gamma);

                // Hàm run giờ sẽ trả về danh sách điểm số của nó
                run_training_parallel(
                    &mut local_env,
                    shared_brain_loop.clone(),
                    pbt_manager.clone(),
                    hot_config.clone(),
                    ep_per_thread,
                    total_target_episodes,
                    current_ep_for_decay,
                    t_id,
                    num_threads,
                    training_policy,
                    BUFF_MULTIPLIER,
                )
            })
            .collect();

        // Gộp tất cả điểm số lại thành 1 list lớn
        let mut all_scores: Vec<f64> = results.into_iter().flatten().collect();

        // ------------------------------------------------------
        // 3. TÍNH TOÁN METRIC (TOP 1% AVG)
        // ------------------------------------------------------
        // Sắp xếp giảm dần để lấy điểm cao nhất
        all_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let top_1_percent_count = (all_scores.len() as f64 * 0.01).ceil() as usize;
        let top_1_percent_count = top_1_percent_count.max(1); // Ít nhất 1
        let top_scores = &all_scores[0..top_1_percent_count];

        let sum_top: f64 = top_scores.iter().sum();
        let current_top1_avg = sum_top / top_1_percent_count as f64;

        let duration = start_time.elapsed();
        println!("\n📊 Stats Loop:");
        println!("   - Max Score: {:.0}", all_scores[0]);
        println!("   - Top 1% Avg (Current): {:.2}", current_top1_avg);
        println!(
            "   - Top 1% Avg (Record):  {:.2}",
            best_stable_brain.best_top1_avg
        );

        // ------------------------------------------------------
        // 4. QUYẾT ĐỊNH: GIỮ HAY RESET?
        // ------------------------------------------------------
        current_global_episode += chunk_episodes;

        // 🔄 SỬA ĐIỀU KIỆN: So sánh với best_stable_brain
        if current_top1_avg > best_stable_brain.best_top1_avg {
            // >>> WIN CASE <<<
            println!("✅ NEW RECORD! Config này ngon. Giữ lại network & config.");

            // 1. Cập nhật Stats vào Brain
            brain.total_episodes += chunk_episodes;
            brain.best_top1_avg = current_top1_avg;

            // 2. Cập nhật biến cục bộ
            current_global_episode = brain.total_episodes;

            // ❌ XÓA DÒNG NÀY (Không cần biến cục bộ nữa):
            // best_top1_percent_avg = current_top1_avg;

            // 3. Cập nhật PBT Config tốt nhất vào Brain (Giữ nguyên)
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

            // 4. LƯU "CHECKPOINT CỨNG" MỚI (Giữ nguyên)
            // Lệnh này sẽ tự động lưu `brain.best_top1_avg` mới vào `best_stable_brain`
            // để vòng lặp sau dùng làm mốc so sánh.
            best_stable_brain = brain.clone();

            // 5. Lưu File (Giữ nguyên)
            let filename = format!("brain_ep_{}.msgpack", current_global_episode);
            if let Err(e) = brain.export_to_msgpack(&filename) {
                eprintln!("❌ Lỗi lưu file: {}", e);
            } else {
                println!("💾 Saved checkpoint: {}", filename);
            }
        } else {
            // >>> LOSE CASE <<<
            println!("❌ FAILED. Hủy bỏ Iteration này.");

            // 🔄 SỬA DÒNG NÀY: Lấy từ best_stable_brain
            println!(
                "🔄 Reverting về trạng thái cũ (Ep: {}, Best: {:.2})",
                best_stable_brain.total_episodes, best_stable_brain.best_top1_avg
            );

            // Revert biến đếm hiển thị bên ngoài cho đúng thực tế
            current_global_episode = best_stable_brain.total_episodes;
        }

        println!("⏱️ Time: {:.1}s | Total Ep: {}\n-----------------------------------------------------------", duration.as_secs_f64(), current_global_episode);
    }
}

// Hàm Watcher (Giữ nguyên)
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

// Sửa hàm run_training_parallel để trả về Vec<f64>
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
    let mut pbt_config = if thread_id == 0 {
        TrainingConfig {
            w_empty: brain.w_empty,
            w_snake: brain.w_snake,
            w_merge: brain.w_merge,
            w_disorder: brain.w_disorder,
        }
    } else {
        TrainingConfig {
            w_empty: (brain.w_empty * rng.random_range(0.9..1.1)),
            w_snake: (brain.w_snake * rng.random_range(0.9..1.1)),
            w_merge: (brain.w_merge * rng.random_range(0.9..1.1)),
            w_disorder: (brain.w_disorder * rng.random_range(0.9..1.1)),
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
