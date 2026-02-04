use rand::Rng; // Nhớ trait này để dùng random_bool
use rayon::prelude::*;
use std::sync::{Arc, Mutex}; // Cần Mutex cho PBT
use threes_rs::{
    n_tuple_network::NTupleNetwork, pbt::PBTManager, pbt::TrainingConfig, python_module::ThreesEnv,
};

// --- (Đoạn Struct SharedBrain giữ nguyên) ---
struct SharedBrain {
    network: *mut NTupleNetwork,
}
unsafe impl Send for SharedBrain {}
unsafe impl Sync for SharedBrain {}

// --- (Đoạn Struct PBTManager và TrainingConfig bác paste vào đây hoặc import) ---
// ... (Code struct PBTManager bác đã có ở bước trước) ...

fn main() {
    let num_episodes = 10_000_000;
    let num_threads = 8;
    let gamma = 0.995;

    // 1. Khởi tạo Brain
    let mut brain = NTupleNetwork::new(0.1, gamma);
    let brain_ptr = SharedBrain {
        network: &mut brain as *mut NTupleNetwork,
    };
    let shared_brain = Arc::new(brain_ptr);

    // 2. KHỞI TẠO PBT MANAGER (Dùng Mutex để các luồng tranh nhau báo cáo)
    // PBTManager::new() là hàm bác đã viết ở bước trước
    let pbt_manager = Arc::new(Mutex::new(PBTManager::new()));

    println!("🚀 Bắt đầu luyện đan PBT với {} luồng...", num_threads);

    (0..num_threads).into_par_iter().for_each(|t_id| {
        let mut local_env = ThreesEnv::new(gamma);
        let ep_per_thread = num_episodes / num_threads as u32;

        run_training_parallel(
            &mut local_env,
            shared_brain.clone(),
            pbt_manager.clone(), // <--- TRUYỀN PBT VÀO
            ep_per_thread,
            num_episodes,
            t_id,
            num_threads,
        );
    });

    brain
        .export_to_binary("brain_super_pbt.dat")
        .expect("Lỗi lưu file");
}

fn run_training_parallel(
    env: &mut ThreesEnv,
    shared_brain: Arc<SharedBrain>,
    pbt: Arc<Mutex<PBTManager>>, // <--- NHẬN PBT
    episodes_to_run: u32,
    total_global_episodes: u32,
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

    for local_ep in 0..episodes_to_run {
        let global_ep = local_ep * num_threads + thread_id;

        // Decay Alpha & Epsilon (Giữ nguyên logic cũ)
        let progress = global_ep as f32 / total_global_episodes as f32;
        let current_alpha = (0.1 * (1.0 - progress)).max(0.001);
        let current_epsilon =
            (0.5 * (1.0 - (global_ep as f64 / (total_global_episodes as f64 * 0.8)))).max(0.01);

        env.reset();

        while !env.game.game_over {
            let action = if rng.random_bool(current_epsilon) {
                env.get_random_valid_action()
            } else {
                env.get_best_action_greedy(brain)
            };

            let (error, _) = env.train_step(brain, action, current_alpha);
            running_error = running_error * 0.9999 + error * 0.0001;
        }

        running_score = running_score * 0.99 + env.game.score as f32 * 0.01;

        // --- PBT CHECKPOINT ---
        // Cứ mỗi 1000 ván, dừng lại để báo cáo và tiến hóa
        if local_ep > 0 && local_ep % 1000 == 0 {
            let mut pbt_guard = pbt.lock().unwrap();

            // Gọi hàm report_and_evolve bác đã viết
            let (evolved, new_cfg) = pbt_guard.report_and_evolve(
                thread_id,
                running_score,
                local_config, // Truyền struct config hiện tại (Copy)
            );

            if evolved {
                // Nếu được lệnh tiến hóa -> Thay đổi Config
                local_config = new_cfg;
                env.set_config(local_config);
                // Mẹo: Reset running_score nhẹ để đo lường config mới khách quan hơn
                // running_score = running_score * 0.5;
            }
        }

        // Log thông tin (Thêm info về config đang dùng)
        if thread_id == 0 && local_ep % 500 == 0 {
            println!(
                "Ep: {:>7} | Err: {:.4} | Score: {:>6.0} | W_Empty: {:.1} | W_Snake: {:.1}",
                global_ep, running_error, running_score, local_config.w_empty, local_config.w_snake
            );
        }

        if thread_id == 0 && global_ep > 0 && global_ep % 200_000 == 0 {
            let _ = brain.export_to_binary(&format!("brain_ep_{}.dat", global_ep));
        }
    }
}
