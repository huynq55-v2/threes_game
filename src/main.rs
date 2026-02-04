use rand::Rng; // Nhớ trait này để dùng random_bool
use rayon::prelude::*;
use std::fs::File; // <--- Thêm File
use std::io::BufReader; // <--- Thêm BufReader
use std::sync::{Arc, Mutex, RwLock}; // Cần Mutex cho PBT
use std::time::Duration;
use std::{env, thread}; // <--- Thêm thread
use threes_rs::hotload_config::HotLoadConfig;
use threes_rs::{
    n_tuple_network::NTupleNetwork, pbt::PBTManager, pbt::TrainingConfig, python_module::ThreesEnv,
}; // <--- Thêm Duration

struct SharedBrain {
    network: *mut NTupleNetwork,
}

#[derive(Clone, Copy, Debug, PartialEq)] // Derive Copy để truyền vào thread không bị move
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

    // Nếu chạy: cargo run -- 2000000
    // Thì nó sẽ tự hiểu là resume từ 2 triệu
    let resume_from_episode = if args.len() > 1 {
        args[1].parse::<usize>().unwrap_or(0) as u32
    } else {
        0 as u32
    };

    // 2. Tham số Policy (Index 2) - MỚI
    let policy_arg = if args.len() > 2 {
        args[2].to_lowercase()
    } else {
        "greedy".to_string() // Mặc định là Greedy nếu không nhập
    };

    let training_policy = match policy_arg.as_str() {
        "expect" | "expectimax" => {
            println!("🧠 Training Mode: EXPECTIMAX (Chậm nhưng chắc)");
            TrainingPolicy::Expectimax
        }
        _ => {
            println!("⚡ Training Mode: GREEDY (Tốc độ bàn thờ)");
            TrainingPolicy::Greedy
        }
    };

    let chunk_episodes = 1_000_000;
    let current_target = resume_from_episode + chunk_episodes;

    println!(
        "🚀 Bắt đầu từ: {} | Mục tiêu đợt này: {}",
        resume_from_episode, current_target
    );

    // Tổng đích đến (để tính Alpha decay cho chuẩn)
    // Ví dụ mục tiêu cuối cùng là 10 triệu
    let total_target_episodes = 100_000_000;

    // --- SỬA LỖI LOADING Ở ĐÂY ---
    let mut brain = if resume_from_episode > 0 {
        let filename = format!("brain_ep_{}.msgpack", resume_from_episode);
        println!("📂 Đang load não từ checkpoint: {}", filename);

        let b = NTupleNetwork::load_from_msgpack(&filename).expect("Không tìm thấy file!");

        // [DEBUG] In ra để xem nó là 0.0 hay là số thực
        println!(
            "🧐 CHECK DATA GỐC: Empty={:.4}, Snake={:.4}",
            b.w_empty, b.w_snake
        );
        b
    } else {
        println!("✨ Tạo não mới tinh...");
        NTupleNetwork::new(0.1, 0.995)
    };
    // -----------------------------

    let brain_ptr = SharedBrain {
        network: &mut brain as *mut NTupleNetwork,
    };
    let shared_brain = Arc::new(brain_ptr);

    // 1. Tạo biến HotConfig chia sẻ
    let hot_config = Arc::new(RwLock::new(HotLoadConfig::default()));

    // 2. Bật Watcher
    start_config_watcher(hot_config.clone());
    println!("👀 Đã bật Hot Reload. Hãy sửa file config.json để can thiệp!");

    // 2. KHỞI TẠO PBT MANAGER (Dùng Mutex để các luồng tranh nhau báo cáo)
    // PBTManager::new() là hàm bác đã viết ở bước trước
    let pbt_manager = Arc::new(Mutex::new(PBTManager::new()));

    println!("🚀 Bắt đầu luyện đan PBT với {} luồng...", num_threads);

    (0..num_threads).into_par_iter().for_each(|t_id| {
        let mut local_env = ThreesEnv::new(gamma);
        let ep_per_thread = chunk_episodes as u32 / num_threads;

        run_training_parallel(
            &mut local_env,
            shared_brain.clone(),
            pbt_manager.clone(),
            hot_config.clone(),
            ep_per_thread,         // Số ván cần chạy đợt này
            total_target_episodes, // Tổng đích (để tính tỷ lệ %)
            resume_from_episode,   // <--- TRUYỀN THÊM OFFSET VÀO
            t_id,
            num_threads,
            training_policy,
        );
    });

    // Save cuối cùng cũng dùng msgpack
    let end_episode = resume_from_episode + chunk_episodes;
    let filename = format!("brain_ep_{}.msgpack", end_episode);
    brain.export_to_msgpack(&filename).expect("Lỗi lưu file");
}

// --- HÀM WATCHER: Chạy ngầm để đọc file json ---
fn start_config_watcher(shared_hot_config: Arc<RwLock<HotLoadConfig>>) {
    thread::spawn(move || {
        loop {
            thread::sleep(Duration::from_secs(2)); // Check mỗi 2 giây

            if let Ok(file) = File::open("config.json") {
                let reader = BufReader::new(file);
                // Parse JSON vào struct
                if let Ok(new_cfg) = serde_json::from_reader(reader) {
                    let mut write_guard = shared_hot_config.write().unwrap();
                    *write_guard = new_cfg;
                }
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
) {
    let mut rng = rand::rng();
    let mut running_error = 0.0;
    let mut running_score = 0.0;

    // 1. LẤY BRAIN TRƯỚC (Để có dữ liệu resume)
    let ptr = shared_brain.network;
    let brain = unsafe { &mut *ptr };

    // 2. KHỞI TẠO PBT_CONFIG (Xử lý cả Resume và Train lần đầu)
    let mut pbt_config = if thread_id == 0 {
        TrainingConfig {
            // Nếu não có giá trị (>0) thì lấy giá trị đó (Resume)
            // Nếu không (lần đầu) thì lấy số mặc định an toàn (ví dụ 40.0 và 50.0)
            w_empty: if brain.w_empty > 0.0 {
                brain.w_empty
            } else {
                50.0
            },
            w_snake: if brain.w_snake > 0.0 {
                brain.w_snake
            } else {
                50.0
            },
        }
    } else {
        // Các thread khác: Nếu não rỗng thì random rộng, nếu có não thì biến động quanh não
        let (base_empty, base_snake) = if brain.w_empty > 0.0 {
            (brain.w_empty, brain.w_snake)
        } else {
            (rng.random_range(30.0..60.0), rng.random_range(20.0..80.0))
        };

        TrainingConfig {
            w_empty: (base_empty * rng.random_range(0.8..1.2)).clamp(1.0, 500.0),
            w_snake: (base_snake * rng.random_range(0.8..1.2)).clamp(0.0, 1000.0),
        }
    };

    // --- VÒNG LẶP CHÍNH ---
    for local_ep in 0..episodes_to_run {
        let current_global_ep = (local_ep * num_threads + thread_id) + start_offset;
        let progress = current_global_ep as f32 / total_target_episodes as f32;

        // 3. XỬ LÝ HOT CONFIG & MERGE
        let current_hot = *hot_config.read().unwrap();
        let mut effective_config = pbt_config;

        if current_hot.w_empty_override > 0.0 {
            effective_config.w_empty = current_hot.w_empty_override;
        }
        if current_hot.w_snake_override > 0.0 {
            effective_config.w_snake = current_hot.w_snake_override;
        }

        // Áp dụng config vào môi trường chơi game
        env.set_config(effective_config);

        // 4. ALPHA & EPSILON DECAY
        let mut current_alpha = (0.1 * (1.0 - progress)).max(0.001);
        if current_hot.alpha_override > 0.0 {
            current_alpha = current_hot.alpha_override;
        }
        let current_epsilon = (0.5 * (1.0 - (progress / 0.8))).max(0.01);

        // 5. TRAINING STEP
        env.reset();
        while !env.game.game_over {
            // Logic chọn nước đi (Action Selection)
            let action = if rng.random_bool(current_epsilon.into()) {
                // Epsilon-Greedy: Vẫn giữ tỷ lệ ngẫu nhiên để khám phá
                env.get_random_valid_action()
            } else {
                // Khai thác (Exploitation) dựa trên Policy đã chọn
                match policy {
                    TrainingPolicy::Greedy => env.get_best_action_greedy(brain),
                    TrainingPolicy::Expectimax => {
                        // Bác cần đảm bảo hàm này đã có trong ThreesEnv nhé!
                        env.get_best_action_expectimax(brain)
                    }
                }
            };

            let (error, _) = env.train_step(brain, action, current_alpha);
            running_error = running_error * 0.9999 + error * 0.0001;
        }
        running_score = running_score * 0.99 + env.game.score as f32 * 0.01;

        // 6. PBT EVOLVE
        if local_ep > 0 && local_ep % 1000 == 0 {
            let mut pbt_guard = pbt.lock().unwrap();
            let (evolved, new_cfg) =
                pbt_guard.report_and_evolve(thread_id, running_score, pbt_config);
            if evolved {
                pbt_config = new_cfg;
            }
        }

        // 7. LOGGING & SAVING (Thread 0 đảm nhiệm)
        if thread_id == 0 {
            // Log mỗi 500 ván của Thread 0 (để theo dõi tiến độ)
            if local_ep % 500 == 0 {
                println!(
                    "Ep: {:>8} | Err: {:.4} | Sc: {:>5.0} | Emp: {:.1} | Snk: {:.1} | Alp: {:.5}",
                    current_global_ep,
                    running_error,
                    running_score,
                    effective_config.w_empty,
                    effective_config.w_snake,
                    current_alpha
                );
            }

            // ĐIỀU KIỆN SAVE: Chỉ save khi chạy xong ván cuối cùng của đợt này
            // local_ep chạy từ 0 đến (episodes_to_run - 1)
            if local_ep == episodes_to_run - 1 {
                // Tính toán con số tổng kết chính xác
                let end_ep_of_chunk = start_offset + (episodes_to_run * num_threads);

                let filename = format!("brain_ep_{}.msgpack", end_ep_of_chunk);

                // Cập nhật config mới nhất vào não để mang đi save
                brain.w_empty = pbt_config.w_empty;
                brain.w_snake = pbt_config.w_snake;

                if let Err(e) = brain.export_to_msgpack(&filename) {
                    eprintln!("❌ Lỗi lưu file: {}", e);
                } else {
                    println!(
                        "💾 [DONE] Đã hoàn thành Chunk. File checkpoint: {}",
                        filename
                    );
                }
            }
        }
    }
}
