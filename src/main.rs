use rand::Rng;
use rayon::prelude::*;
use std::sync::Arc;
use threes_rs::{n_tuple_network::NTupleNetwork, python_module::ThreesEnv};

// Cấu trúc Wrapper để cho phép chia sẻ Brain qua các luồng mà không cần Lock
struct SharedBrain {
    network: *mut NTupleNetwork,
}

unsafe impl Send for SharedBrain {}
unsafe impl Sync for SharedBrain {}

// fn main() {
//     let num_episodes = 10_000_000; // Nâng lên 5 triệu ván để AI thành tinh
//     let num_threads = 8; // Lấy số luồng tối đa của CPU

//     let mut brain = NTupleNetwork::new(0.1, 0.99);
//     let brain_ptr = SharedBrain {
//         network: &mut brain as *mut NTupleNetwork,
//     };
//     let shared_brain = Arc::new(brain_ptr);

//     println!("Bắt đầu luyện đan với {} luồng...", num_threads);

//     // Chia tổng số episode cho các luồng
//     (0..num_threads).into_par_iter().for_each(|t_id| {
//         let mut local_env = ThreesEnv::new();
//         let episodes_per_thread = num_episodes / num_threads as u32;

//         // Copy cái Arc để đưa vào luồng
//         let thread_brain = shared_brain.clone();

//         run_training_parallel(
//             &mut local_env,
//             thread_brain, // Truyền thẳng vào
//             episodes_per_thread,
//             num_episodes,
//             t_id as u32,
//         );
//     });

//     // Sau khi các luồng chạy xong, lưu bản cuối cùng
//     brain
//         .export_to_binary("brain_final.dat")
//         .expect("Lỗi lưu file");
//     println!("Training Finished. Final brain saved.");
// }

fn main() {
    let num_episodes = 10_000_000; // Máy tính thì cứ quất 10 triệu ván
    let num_threads = 8;

    // 1. Khởi tạo não mới (vì kiến trúc thay đổi)
    // Dùng Alpha cao 0.1 để học nhanh giai đoạn đầu
    let mut brain = NTupleNetwork::new(0.1, 0.995);

    // 2. Bọc vào SharedBrain để chạy đa luồng Hogwild!
    let brain_ptr = SharedBrain {
        network: &mut brain as *mut NTupleNetwork,
    };
    let shared_brain = Arc::new(brain_ptr);

    println!(
        "Bắt đầu luyện đan Cảnh Giới Mới với {} luồng...",
        num_threads
    );

    (0..num_threads).into_par_iter().for_each(|t_id| {
        // Mỗi luồng cần một Env (bàn cờ) riêng
        let mut local_env = ThreesEnv::new();
        let ep_per_thread = num_episodes / num_threads as u32;

        run_training_parallel(
            &mut local_env,
            shared_brain.clone(),
            ep_per_thread,
            num_episodes,
            t_id as u32,
        );
    });

    brain
        .export_to_binary("brain_super_v1.dat")
        .expect("Lỗi lưu file");
}

fn run_training_parallel(
    env: &mut ThreesEnv,
    shared_brain: Arc<SharedBrain>, // Không cần mut ở đây
    episodes_to_run: u32,
    total_global_episodes: u32,
    thread_id: u32,
) {
    let mut rng = rand::rng();
    let mut running_error = 0.0;
    let mut running_score = 0.0;

    // Lấy con trỏ thô ra trước
    let ptr = shared_brain.network;

    // Ép kiểu con trỏ thô thành tham chiếu mutable độc lập
    // Việc này "ngắt" sự liên kết trực tiếp với biến shared_brain của Rust
    let brain = unsafe { &mut *ptr };

    for local_ep in 0..episodes_to_run {
        // Tính episode toàn cục để tính Decay chính xác
        let global_ep = local_ep * 8 as u32 + thread_id;

        // Decay Alpha & Epsilon
        brain.alpha = (0.1 * (1.0 - (global_ep as f32 / total_global_episodes as f32))).max(0.001);
        let epsilon =
            (0.5 * (1.0 - (global_ep as f64 / (total_global_episodes as f64 * 0.8)))).max(0.01);

        env.reset();
        while !env.game.game_over {
            let action = if rng.random_bool(epsilon) {
                env.get_random_valid_action()
            } else {
                // Khi train đa luồng, dùng Predict cho nhanh,
                // hoặc Expectimax nếu CPU bác đủ mạnh
                env.get_best_action_expectimax(brain)
            };

            let (error, _) = env.train_step(brain, action, global_ep, total_global_episodes);
            running_error = running_error * 0.9999 + error * 0.0001;
        }

        running_score = running_score * 0.99 + env.game.score as f32 * 0.01;

        // Chỉ luồng 0 mới in log để tránh loạn màn hình
        if thread_id == 0 && local_ep % 500 == 0 {
            println!(
                "Global Ep: {:>7} | Alpha: {:.4} | AvgErr: {:>8.4} | AvgScore: {:>8.1}",
                global_ep, brain.alpha, running_error, running_score
            );
        }

        // Luồng 0 thực hiện checkpoint
        if thread_id == 0 && global_ep > 0 && global_ep % 200_000 == 0 {
            let _ = brain.export_to_binary(&format!("brain_ep_{}.dat", global_ep));
        }
    }
}
