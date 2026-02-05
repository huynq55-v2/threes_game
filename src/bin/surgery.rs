use std::env;
use std::process;
use threes_rs::n_tuple_network::NTupleNetwork;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        println!("🏥 PHẪU THUẬT TẨY TỦY (Surgery Tool)");
        println!("--------------------------------------");
        println!("CÁCH DÙNG: cargo run --bin surgery <tên_file_brain> <mức_điểm_muốn_đặt>");
        println!("Ví dụ: cargo run --bin surgery brain_ep_1830000.msgpack 3000");
        process::exit(1);
    }

    let filename = &args[1];
    let new_record: f64 = args[2]
        .parse()
        .expect("❌ Mức điểm phải là một số thực (f64)");

    println!("💉 Đang tiến hành phẫu thuật file: {} ...", filename);

    // 1. Load não hiện tại
    let mut brain = match NTupleNetwork::load_from_msgpack(filename) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("❌ Không thể mở file: {}", e);
            process::exit(1);
        }
    };

    println!("📊 Thông số cũ:");
    println!("   - Episode: {}", brain.total_episodes);
    println!("   - Kỷ lục cũ (Top 1%): {:.2}", brain.best_top1_avg);
    println!("   - Overall Avg: {:.2}", brain.best_overall_avg);
    println!("   - Bottom 10% Avg: {:.2}", brain.best_bot10_avg);

    // 2. Thực hiện tẩy tủy
    println!("--------------------------------------");
    println!("🔪 Đang hạ thấp tiêu chuẩn xuống: {:.2}", new_record);

    // Đồng bộ các chỉ số khác xuống mức thấp hơn để AI dễ dàng "Win" vòng đầu tiên
    brain.best_overall_avg = new_record;

    // 3. Lưu lại
    let output_filename = format!("cured_{}", filename);
    match brain.export_to_msgpack(&output_filename) {
        Ok(_) => {
            println!("✅ PHẪU THUẬT THÀNH CÔNG!");
            println!("💾 File mới đã lưu: {}", output_filename);
            println!("🚀 Bây giờ bác hãy dùng file này để Resume Training.");
            println!("💡 AI sẽ dễ dàng đạt NEW RECORD và cập nhật Weights mới.");
        }
        Err(e) => eprintln!("❌ Lỗi khi lưu file: {}", e),
    }
}
