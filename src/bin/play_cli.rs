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
const BUFF_MULTIPLIER: f64 = 10.0 * GOLDEN_RATIO; // ~16.18

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

    // Episode bắt đầu (Resume)
    let mut current_global_episode = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(0) as u32
    } else {
        0 as u32
    };

    // Policy
    let policy_arg = if args.len() > 2 {
        args[2].to_lowercase()
    } else {
        "expect".to_string() // Mặc định Expectimax theo ý bác
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

    // --- SETUP BRAIN ---
    let mut brain = if current_global_episode > 0 {
        let filename = format!("brain_ep_{}.msgpack", current_global_episode);
        println!("📂 Loading brain: {}", filename);
        let b = NTupleNetwork::load_from_msgpack(&filename).expect("Không tìm thấy file!");
        println!(
            "🧐 LOAD DATA: E={:.1}, S={:.1}, M={:.1}, D={:.1}",
            b.w_empty, b.w_snake, b.w_merge, b.w_disorder
        );
        b
    } else {
        println!("✨ Tạo não mới tinh...");
        NTupleNetwork::new(0.1, 0.995)
    };

    // Đảm bảo brain có giá trị khởi tạo hợp lý nếu load từ file cũ chưa có merge/disorder
    if brain.w_empty == 0.0 {
        brain.w_empty = 50.0;
    }
    if brain.w_snake == 0.0 {
        brain.w_snake = 50.0;
    }
    if brain.w_merge == 0.0 {
        brain.w_merge = 15.0;
    }
    if brain.w_disorder == 0.0 {
        brain.w_disorder = 5.0;
    }

    // Pointer unsafe (Code cũ của bác)
    let brain_ptr = SharedBrain {
        network: &mut brain as *mut NTupleNetwork,
    };
    let shared_brain = Arc::new(brain_ptr);

    // Watcher Config
    let hot_config = Arc::new(RwLock::new(HotLoadConfig::default()));
    start_config_watcher(hot_config.clone());

    // PBT Manager
    let pbt_manager = Arc::new(Mutex::new(PBTManager::new()));

    // Số ván mỗi Iteration (Turn)
    let chunk_episodes = 100_000; // 100k ván mỗi vòng lặp
    let total_target_episodes = 100_000_000; // Target ảo để tính decay

    println!(
        "🚀 Bắt đầu Continuous Training với {} luồng...",
        num_threads
    );

    // ==========================================================
    // VÒNG LẶP VĨNH CỬU (CONTINUOUS TRAINING LOOP)
    // ==========================================================
    loop {
        let start_time = std::time::Instant::now();

        // ------------------------------------------------------
        // 1. LOGIC BUFF (Random 1 chỉ số nhân với 16.18)
        // ------------------------------------------------------
        let mut rng = rand::rng();
        let buff_idx = rng.random_range(0..4);

        // Lưu giá trị cũ để log cho dễ nhìn
        let old_vals = (
            brain.w_empty,
            brain.w_snake,
            brain.w_merge,
            brain.w_disorder,
        );

        match buff_idx {
            0 => {
                brain.w_empty *= BUFF_MULTIPLIER;
                // Clamp lại kẻo to quá
                // brain.w_empty = brain.w_empty.clamp(1.0, 5000.0);
                print!("✨ BUFF EMPTY! ");
            }
            1 => {
                brain.w_snake *= BUFF_MULTIPLIER;
                // brain.w_snake = brain.w_snake.clamp(1.0, 5000.0);
                print!("🐍 BUFF SNAKE! ");
            }
            2 => {
                brain.w_merge *= BUFF_MULTIPLIER;
                // brain.w_merge = brain.w_merge.clamp(1.0, 2000.0);
                print!("🔗 BUFF MERGE! ");
            }
            _ => {
                brain.w_disorder *= BUFF_MULTIPLIER;
                // brain.w_disorder = brain.w_disorder.clamp(1.0, 1000.0);
                print!("⚡ BUFF DISORDER! ");
            }
        }

        println!(
            "(E:{:.1}, S:{:.1}, M:{:.1}, D:{:.1}) -> Buffed to {:.1}",
            old_vals.0,
            old_vals.1,
            old_vals.2,
            old_vals.3,
            match buff_idx {
                0 => brain.w_empty,
                1 => brain.w_snake,
                2 => brain.w_merge,
                _ => brain.w_disorder,
            }
        );

        // ------------------------------------------------------
        // 2. CHẠY SONG SONG (PARALLEL EXECUTION)
        // ------------------------------------------------------
        let ep_per_thread = chunk_episodes as u32 / num_threads;

        (0..num_threads).into_par_iter().for_each(|t_id| {
            let mut local_env = ThreesEnv::new(gamma);

            run_training_parallel(
                &mut local_env,
                shared_brain.clone(),
                pbt_manager.clone(),
                hot_config.clone(),
                ep_per_thread,
                total_target_episodes,
                current_global_episode, // Offset hiện tại
                t_id,
                num_threads,
                training_policy,
            );
        });

        // ------------------------------------------------------
        // 3. TỔNG KẾT & LƯU FILE
        // ------------------------------------------------------
        current_global_episode += chunk_episodes; // Cộng dồn số ván đã chạy

        // Lấy PBT config tốt nhất hiện tại gán ngược lại vào não chính để lưu
        // Lưu ý: PBT Manager giữ config tốt nhất trong RAM
        {
            let pbt = pbt_manager.lock().unwrap();
            // Lấy thằng tốt nhất (đã sort trong quá trình chạy)
            // Đây là mẹo: PBT lưu population, ta lấy thằng best trong đó
            if let Some(best_thread) = pbt.get_best_config_entry() {
                let best_cfg = best_thread.1;
                brain.w_empty = best_cfg.w_empty;
                brain.w_snake = best_cfg.w_snake;
                brain.w_merge = best_cfg.w_merge;
                brain.w_disorder = best_cfg.w_disorder;
                println!(
                    "🏆 Winner Config: E:{:.1} S:{:.1} M:{:.1} D:{:.1}",
                    brain.w_empty, brain.w_snake, brain.w_merge, brain.w_disorder
                );
            }
        }

        let filename = format!("brain_ep_{}.msgpack", current_global_episode);
        if let Err(e) = brain.export_to_msgpack(&filename) {
            eprintln!("❌ Lỗi lưu file: {}", e);
        } else {
            let duration = start_time.elapsed();
            println!(
                "💾 [DONE] Checkpoint: {} (Time: {:.1}s)",
                filename,
                duration.as_secs_f64()
            );
            println!("-----------------------------------------------------------");
        }
    }
}

// Hàm Watcher giữ nguyên
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

// Hàm chạy Parallel (Sửa lại chút xíu ở đoạn lấy Config ban đầu)
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
) {
    let mut rng = rand::rng();
    let mut running_error = 0.0;
    let mut running_score = 0.0;

    // LẤY NÃO (UNSAFE)
    let ptr = shared_brain.network;
    let brain = unsafe { &mut *ptr };

    // KHỞI TẠO CONFIG CHO THREAD NÀY
    // Thread 0: Lấy đúng giá trị từ não (đã được Buff ở main)
    // Thread khác: Biến động nhẹ quanh giá trị Buff đó
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
        let mut current_alpha = (0.01 * (1.0 - progress)).max(0.0001); // Giảm alpha đi chút cho expectimax
        if current_hot.alpha_override > 0.0 {
            current_alpha = current_hot.alpha_override;
        }
        let current_epsilon = (0.2 * (1.0 - (progress / 0.8))).max(0.01); // Expectimax thì ít cần epsilon cao

        // GAME LOOP
        env.reset();

        let mut step_count = 0;
        while !env.game.game_over {
            step_count += 1;
            if step_count > 20000 {
                break;
            } // Chống treo

            let action = if rng.random_bool(current_epsilon.into()) {
                env.get_random_valid_action()
            } else {
                match policy {
                    TrainingPolicy::Greedy => env.get_best_action_greedy(brain),
                    TrainingPolicy::Expectimax => env.get_best_action_expectimax(brain),
                }
            };

            // Fix lỗi loop 0 (nếu action=0 mà ko đi được) -> get_best_action nên trả về valid move
            // Ở đây tạm tin tưởng hàm get_best_action của bác đã check valid
            let (error, _) = env.train_step(brain, action, current_alpha);
            running_error = running_error * 0.999 + error * 0.001;
        }
        running_score = running_score * 0.99 + env.game.score as f64 * 0.01;

        // PBT EVOLVE
        if local_ep > 0 && local_ep % 1000 == 0 {
            let mut pbt_guard = pbt.lock().unwrap();
            let (evolved, new_cfg) =
                pbt_guard.report_and_evolve(thread_id, running_score, pbt_config);
            if evolved {
                pbt_config = new_cfg;
            }
        }

        // LOGGING (Chỉ Thread 0 log)
        if thread_id == 0 && local_ep % 1000 == 0 {
            print!(
                "\r   Run: {:>6} | Sc: {:>5.0} | Cfg: S{:.0} M{:.0}   ",
                local_ep, running_score, effective_config.w_snake, effective_config.w_merge
            );
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
    }
    // Xuống dòng khi xong chunk của thread 0
    if thread_id == 0 {
        println!();
    }
}
