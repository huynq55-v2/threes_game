use std::env;
use threes_rs::n_tuple_network::NTupleNetwork;

fn main() {
    // Tên file Huy đang có
    let input_file = "brain_ep_3760000_old.msgpack";
    let output_file = "brain_ep_3760000.msgpack";

    println!("📂 Đang nạp bộ não cũ: {}...", input_file);

    // 1. Load bộ não hiện tại
    let mut brain = match NTupleNetwork::load_from_msgpack(input_file) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("❌ Không tìm thấy file hoặc lỗi định dạng: {}", e);
            return;
        }
    };

    let old_tables = brain.weights.len();
    let old_tuples = brain.tuples.len();

    println!(
        "🧠 Trạng thái cũ: {} bảng weights, {} tuples.",
        old_tables, old_tuples
    );

    // 2. Cấy ghép thêm 9 ô vuông 2x2
    add_all_2x2_squares(&mut brain);

    let new_tables = brain.weights.len();
    let new_tuples = brain.tuples.len();

    println!(
        "✨ Trạng thái mới: {} bảng weights (+{}), {} tuples (+{}).",
        new_tables,
        new_tables - old_tables,
        new_tuples,
        new_tuples - old_tuples
    );

    // 3. Xuất ra file mới
    match brain.export_to_msgpack(output_file) {
        Ok(_) => println!("💾 Đã lưu bộ não nâng cấp thành công vào: {}", output_file),
        Err(e) => eprintln!("❌ Lỗi khi lưu file: {}", e),
    }

    println!("\n🚀 XONG! Giờ Huy có thể dùng file v2 này để tiếp tục huấn luyện.");
    println!("💡 Lưu ý: Đừng quên tăng Alpha lên một chút trong vài iter đầu để AI 'khai phá' các ô vuông mới này.");
}

/// Hàm logic cấy ghép 9 ô vuông 2x2
fn add_all_2x2_squares(brain: &mut NTupleNetwork) {
    let table_size = 15usize.pow(4); // 2x2 = 4 ô

    // Nhóm 1: 4 Góc (Corners) -> Gốc: [0, 1, 4, 5]
    brain.weights.push(vec![0.0; table_size]);
    let id_corner = brain.weights.len() - 1;
    add_symmetries_shared_manual(brain, vec![0, 1, 4, 5], id_corner);

    // Nhóm 2: 4 Cạnh (Edge-Middles) -> Gốc: [1, 2, 5, 6]
    brain.weights.push(vec![0.0; table_size]);
    let id_edge = brain.weights.len() - 1;
    add_symmetries_shared_manual(brain, vec![1, 2, 5, 6], id_edge);

    // Nhóm 3: 1 Trung tâm (Center) -> Gốc: [5, 6, 9, 10]
    brain.weights.push(vec![0.0; table_size]);
    let id_center = brain.weights.len() - 1;
    add_symmetries_shared_manual(brain, vec![5, 6, 9, 10], id_center);
}

/// Hàm bổ trợ để sinh đối xứng cho Tuple mới (Copy logic từ NTupleNetwork của Huy)
fn add_symmetries_shared_manual(
    brain: &mut NTupleNetwork,
    base_tuple: Vec<usize>,
    weight_id: usize,
) {
    let rotate = |idx: usize| -> usize {
        let r = idx / 4;
        let c = idx % 4;
        c * 4 + (3 - r)
    };
    let mirror = |idx: usize| -> usize {
        let r = idx / 4;
        let c = idx % 4;
        r * 4 + (3 - c)
    };

    let mut variants = Vec::new();
    let mut current_tuple = base_tuple;

    for _ in 0..4 {
        variants.push(current_tuple.clone());
        let mirrored: Vec<usize> = current_tuple.iter().map(|&x| mirror(x)).collect();
        variants.push(mirrored);
        current_tuple = current_tuple.iter().map(|&x| rotate(x)).collect();
    }

    variants.sort();
    variants.dedup();

    for v in variants {
        // Huy cần đảm bảo struct TupleConfig có thể truy cập được (pub)
        brain.tuples.push(threes_rs::n_tuple_network::TupleConfig {
            indices: v,
            weight_index: weight_id,
        });
    }
}
