use rand::Rng; // Nhớ trait này để dùng random_bool
use rayon::prelude::*;
use std::fs::File; // <--- Thêm File
use std::io::BufReader; // <--- Thêm BufReader
use std::sync::{Arc, Mutex, RwLock}; // Cần Mutex cho PBT
use std::thread; // <--- Thêm thread
use std::time::Duration;
use threes_rs::hotload_config::HotLoadConfig;
use threes_rs::{
    n_tuple_network::NTupleNetwork, pbt::PBTManager, pbt::TrainingConfig, python_module::ThreesEnv,
}; // <--- Thêm Duration

struct SharedBrain {
    network: *mut NTupleNetwork,
}
unsafe impl Send for SharedBrain {}
unsafe impl Sync for SharedBrain {}

fn main() {
    let num_threads = 8;

    let gamma = 0.995;

    // CẤU HÌNH CHẠY TỪNG KHÚC (CHUNK)
    let chunk_episodes = 1_000_000; // Mỗi lần chạy 1 triệu ván rồi nghỉ

    // QUAN TRỌNG: Lần 1 để bằng 0.
    // Lần 2 (khi đã có file brain_ep_1000000.dat) thì sửa thành 1_000_000
    let resume_from_episode = 0;

    // Tổng đích đến (để tính Alpha decay cho chuẩn)
    // Ví dụ mục tiêu cuối cùng là 10 triệu
    let total_target_episodes = 100_000_000;

    // --- SỬA LỖI LOADING Ở ĐÂY ---
    let mut brain = if resume_from_episode > 0 {
        let filename = format!("brain_ep_{}.msgpack", resume_from_episode); // Đổi đuôi .msgpack
        println!("📂 Đang load não từ checkpoint: {}", filename);

        // Gọi hàm mới load_from_msgpack
        NTupleNetwork::load_from_msgpack(&filename).expect("Không tìm thấy file não để load!")
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
        let ep_per_thread = chunk_episodes / num_threads as u32;

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
    start_offset: u32, // <--- THAM SỐ MỚI
    thread_id: u32,
    num_threads: u32,
) {
    let mut rng = rand::rng();
    let mut running_error = 0.0;
    let mut running_score = 0.0;

    // --- PBT SETUP: KHỞI TẠO CONFIG ---
    // Thread 0: Giữ config chuẩn (Baseline)
    // Các thread khác: Random để tìm vùng đất mới
    let mut local_config = if thread_id == 0 {
        TrainingConfig {
            w_empty: 50.0,
            // w_disorder: 1.0,
            w_snake: 0.0,
        }
    } else {
        TrainingConfig {
            // Random w_empty từ 30 -> 80
            w_empty: rng.random_range(30.0..80.0),
            // Random w_disorder từ 0.5 -> 2.0
            // w_disorder: rng.random_range(0.5..2.0),
            w_snake: 0.0,
        }
    };

    // Áp dụng config ngay lập tức
    env.set_config(local_config);

    // Hogwild Magic
    let ptr = shared_brain.network;
    let brain = unsafe { &mut *ptr };

    // Config riêng của Thread này (Do PBT quản lý)
    let mut pbt_config = TrainingConfig::default();
    if thread_id != 0 {
        // Random khởi tạo để đa dạng hóa quần thể
        pbt_config.w_empty = rng.gen_range(30.0..80.0);
        pbt_config.w_snake = rng.gen_range(0.0..0.5);
    }

    // --- VÒNG LẶP CHÍNH ---
    for local_ep in 0..episodes_to_run {
        // 1. TÍNH TOÁN TIẾN ĐỘ
        let current_global_ep = (local_ep * num_threads + thread_id) + start_offset;
        let progress = current_global_ep as f32 / total_target_episodes as f32;

        // 2. XỬ LÝ HOT CONFIG (Ưu tiên file config.json)
        // Đọc cấu hình từ file (Read Lock - rất nhanh)
        let current_hot = *hot_config.read().unwrap();

        // Merge: Nếu file có set (>0) thì dùng file, không thì dùng PBT
        let mut effective_config = pbt_config;
        if current_hot.w_empty_override > 0.0 {
            effective_config.w_empty = current_hot.w_empty_override;
        }
        if current_hot.w_snake_override > 0.0 {
            effective_config.w_snake = current_hot.w_snake_override;
        }

        // Áp dụng vào môi trường
        env.set_config(effective_config);

        // 3. TÍNH ALPHA & EPSILON
        let mut current_alpha = (0.1 * (1.0 - progress)).max(0.001);
        // Nếu file ép buộc Alpha
        if current_hot.alpha_override > 0.0 {
            current_alpha = current_hot.alpha_override;
        }

        let current_epsilon = (0.5 * (1.0 - (progress / 0.8))).max(0.01);

        // 4. CHƠI GAME (Training Loop)
        env.reset();
        while !env.game.game_over {
            let action = if rng.random_bool(current_epsilon.into()) {
                // Hoặc random_bool nếu dùng rand 0.9
                env.get_random_valid_action()
            } else {
                env.get_best_action_greedy(brain)
            };

            let (error, _) = env.train_step(brain, action, current_alpha);
            running_error = running_error * 0.9999 + error * 0.0001;
        }
        running_score = running_score * 0.99 + env.game.score as f32 * 0.01;

        // 5. PBT EVOLVE (Mỗi 1000 ván)
        if local_ep > 0 && local_ep % 1000 == 0 {
            let mut pbt_guard = pbt.lock().unwrap();
            // Báo cáo config GỐC (pbt_config) chứ không phải config đã merge
            let (evolved, new_cfg) =
                pbt_guard.report_and_evolve(thread_id, running_score, pbt_config);

            if evolved {
                pbt_config = new_cfg; // Cập nhật config gốc
                                      // Reset điểm nhẹ để đo lường config mới
                                      // running_score *= 0.9;
            }
        }

        // 6. LOGGING (Chỉ Thread 0)
        if thread_id == 0 && local_ep % 500 == 0 {
            // Helper in ra xem có đang Override không
            let fmt = |val: f32, ovr: f32| {
                if ovr > 0.0 {
                    format!("{:.1}(F)", ovr)
                } else {
                    format!("{:.1}", val)
                }
            };

            println!(
                "Ep: {:>8} | Err: {:.4} | Sc: {:>5.0} | Emp: {} | Snk: {} | Alp: {:.5}",
                current_global_ep,
                running_error,
                running_score,
                fmt(pbt_config.w_empty, current_hot.w_empty_override),
                fmt(pbt_config.w_snake, current_hot.w_snake_override),
                current_alpha
            );
        }

        // 7. SAVE CHECKPOINT (MessagePack)
        if thread_id == 0 && current_global_ep > 0 && current_global_ep % 1_000_000 == 0 {
            let filename = format!("brain_ep_{}.msgpack", current_global_ep); // Đổi đuôi file cho dễ nhớ
            if let Err(e) = brain.export_to_msgpack(&filename) {
                eprintln!("❌ Lỗi lưu file {}: {}", filename, e);
            } else {
                println!("💾 Saved Android-ready model: {}", filename);
            }
        }
    }
}
