use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
};

use serde::{Deserialize, Serialize};

use rmp_serde::{Deserializer, Serializer};

use crate::game::Game;

#[derive(Serialize, Deserialize)]
pub struct TupleConfig {
    pub indices: Vec<usize>, // Các ô trên bàn cờ (ví dụ: [0,1,2,3,7])
    pub weight_index: usize, // Trỏ đến bảng weights số mấy (ví dụ: bảng số 0)
}

#[derive(Serialize, Deserialize)]
pub struct NTupleNetwork {
    pub tuples: Vec<TupleConfig>, // Danh sách 96 con rắn
    pub weights: Vec<Vec<f32>>,   // Chỉ có 12 bảng dữ liệu thôi
    pub alpha: f32,
    pub gamma: f32,

    pub w_empty: f32,
    pub w_snake: f32,
    pub w_disorder: f32,
}

impl NTupleNetwork {
    pub fn new(alpha: f32, gamma: f32) -> Self {
        let mut network = NTupleNetwork {
            tuples: Vec::new(),
            weights: Vec::new(),
            alpha,
            gamma,
            w_empty: 0.0,
            w_snake: 0.0,
            w_disorder: 0.0,
        };

        network.add_shared_snake();

        // network.init_weights();
        network
    }

    pub fn export_to_msgpack(&self, filename: &str) -> std::io::Result<()> {
        let file = File::create(filename)?;
        let mut writer = BufWriter::new(file);

        // --- SỬA DÒNG NÀY ---
        // Thêm .with_struct_map() để ép ghi tên trường (Map) thay vì thứ tự (Array)
        let mut serializer = Serializer::new(&mut writer).with_struct_map();

        self.serialize(&mut serializer)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

        Ok(())
    }

    // Hàm load để Resume training
    pub fn load_from_msgpack(filename: &str) -> std::io::Result<Self> {
        let file = File::open(filename)?;
        let reader = BufReader::new(file);
        let mut deserializer = Deserializer::new(reader);

        let network: NTupleNetwork = Deserialize::deserialize(&mut deserializer)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        Ok(network)
    }

    fn add_symmetries_shared(&mut self, base_tuple: Vec<usize>, weight_id: usize) {
        // Định nghĩa closure để Xoay 90 độ theo chiều kim đồng hồ
        // Công thức: (row, col) -> (col, 3 - row)
        let rotate = |idx: usize| -> usize {
            let r = idx / 4;
            let c = idx % 4;
            c * 4 + (3 - r)
        };

        // Định nghĩa closure để Lật Ngang (Mirror Horizontal)
        // Công thức: (row, col) -> (row, 3 - col)
        let mirror = |idx: usize| -> usize {
            let r = idx / 4;
            let c = idx % 4;
            r * 4 + (3 - c)
        };

        let mut variants = Vec::new();
        let mut current_tuple = base_tuple;

        // Sinh ra 4 góc xoay (0, 90, 180, 270)
        for _ in 0..4 {
            // 1. Thêm bản xoay hiện tại
            variants.push(current_tuple.clone());

            // 2. Thêm bản lật gương của bản xoay hiện tại
            // (Map từng phần tử qua hàm mirror)
            let mirrored: Vec<usize> = current_tuple.iter().map(|&x| mirror(x)).collect();
            variants.push(mirrored);

            // 3. Xoay tuple 90 độ để chuẩn bị cho vòng lặp tiếp theo
            current_tuple = current_tuple.iter().map(|&x| rotate(x)).collect();
        }

        // Lọc trùng (Deduplication)
        // Cần thiết vì một số pattern đối xứng (như hình vuông ở giữa)
        // khi xoay/lật sẽ tạo ra các index y hệt nhau.
        variants.sort();
        variants.dedup();

        // Đẩy vào danh sách Tuples chính thức
        for v in variants {
            self.tuples.push(TupleConfig {
                indices: v,
                weight_index: weight_id, // Tất cả biến thể đều trỏ về cùng 1 bảng weights
            });
        }
    }

    pub fn add_shared_snake(&mut self) {
        let snake_path = vec![0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12];

        // Sliding Window 5 ô
        for i in 0..=(snake_path.len() - 5) {
            // 1. KHỞI TẠO WEIGHTS NGAY TẠI ĐÂY (Master)
            let table_size = 15usize.pow(5);
            self.weights.push(vec![0.0; table_size]);
            let current_weight_id = self.weights.len() - 1;

            // 2. Tạo Tuple gốc
            let mut base_indices = Vec::new();
            for j in 0..5 {
                base_indices.push(snake_path[i + j]);
            }

            // 3. Sinh 8 biến thể (Slaves) trỏ về Master Weight này
            self.add_symmetries_shared(base_indices, current_weight_id);
        }
    }

    pub fn encode_tile(value: u32) -> usize {
        if value == 0 {
            return 0;
        }
        if value == 1 {
            return 1;
        }
        if value == 2 {
            return 2;
        }
        let code = ((value as f32 / 3.0).log2() as usize) + 3;
        code.min(14)
    }

    pub fn predict(&self, board: &[u32; 16]) -> f32 {
        let mut sum = 0.0;

        // Tính sẵn code cho cả bàn cờ để nhanh
        let mut encoded_board = [0usize; 16];
        for i in 0..16 {
            encoded_board[i] = Self::encode_tile(board[i]);
        }

        for tuple in &self.tuples {
            let mut idx = 0;
            // Tính index dựa trên vị trí của Tuple này (Slaves)
            for &pos in &tuple.indices {
                idx = idx * 15 + encoded_board[pos];
            }

            // Nhưng lấy điểm từ bảng chung (Master)
            sum += self.weights[tuple.weight_index][idx];
        }

        sum
    }

    pub fn predict_game(&self, game: &Game) -> f32 {
        let mut board_flat = [0u32; 16];
        for r in 0..4 {
            for c in 0..4 {
                board_flat[r * 4 + c] = game.board[r][c].value;
            }
        }
        self.predict(&board_flat)
    }

    pub fn update_weights(&mut self, board: &[u32; 16], delta: f32) {
        // Delta đã được chia nhỏ từ bên ngoài

        let mut encoded_board = [0usize; 16];
        for i in 0..16 {
            encoded_board[i] = Self::encode_tile(board[i]);
        }

        for tuple in &self.tuples {
            let mut idx = 0;
            for &pos in &tuple.indices {
                idx = idx * 15 + encoded_board[pos];
            }

            // Cập nhật vào bảng chung
            // Lưu ý: Nhiều tuple sẽ cùng cộng dồn vào 1 bảng này -> Học siêu nhanh
            self.weights[tuple.weight_index][idx] += delta;
        }
    }
}
