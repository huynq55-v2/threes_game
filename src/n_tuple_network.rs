use std::{
    fs::File,
    io::{BufReader, BufWriter},
};

use serde::{Deserialize, Serialize};

use rmp_serde::{Deserializer, Serializer};

use crate::game::Game;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TupleConfig {
    pub indices: Vec<usize>, // Các ô trên bàn cờ (ví dụ: [0,1,2,3,7])
    pub weight_index: usize, // Trỏ đến bảng weights số mấy (ví dụ: bảng số 0)
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NTupleNetwork {
    pub tuples: Vec<TupleConfig>,
    pub weights: Vec<Vec<f64>>,
    pub alpha: f64,
    pub gamma: f64,

    #[serde(default)]
    pub w_empty: f64,
    #[serde(default)]
    pub w_snake: f64,
    #[serde(default)]
    pub w_disorder: f64,
    #[serde(default)]
    pub w_merge: f64,

    // --- THÊM 2 TRƯỜNG NÀY ---
    #[serde(default)]
    pub total_episodes: u32, // Lưu tổng số ván đã train (để tính alpha/epsilon)
    #[serde(default)]
    pub best_top1_avg: f64, // Lưu kỷ lục điểm số
    // THÊM 2 DÒNG NÀY:
    #[serde(default)]
    pub best_overall_avg: f64,
    #[serde(default)]
    pub best_bot10_avg: f64,

    // --- CÁC TRƯỜNG CHO TD(LAMBDA) ---
    #[serde(default = "default_lambda")]
    pub lambda: f64,

    // Dùng skip để không lưu vào file save (giảm dung lượng)
    #[serde(skip)]
    pub traces: Vec<Vec<f64>>,

    // Tối ưu: Chỉ lưu index của những ô có trace > 0
    #[serde(skip)]
    pub active_trace_indices: Vec<Vec<usize>>,

    #[serde(default = "default_phase")]
    pub phase: bool, // True = Tăng (Golden Ratio), False = Giảm (1/Golden Ratio)
}

fn default_lambda() -> f64 {
    0.9
}

fn default_phase() -> bool {
    true
}

impl NTupleNetwork {
    pub fn new(alpha: f64, gamma: f64) -> Self {
        let mut network = NTupleNetwork {
            tuples: Vec::new(),
            weights: Vec::new(),
            alpha,
            gamma,
            w_empty: 0.0,
            w_snake: 0.0,
            w_disorder: 0.0,
            w_merge: 0.0,

            // Khởi tạo mặc định
            total_episodes: 0,
            best_top1_avg: 0.0,
            best_overall_avg: 0.0,
            best_bot10_avg: 0.0,

            // Default params
            lambda: 0.9,
            traces: Vec::new(),
            active_trace_indices: Vec::new(),
            phase: true,
        };

        network.add_shared_snake();

        // QUAN TRỌNG: Đồng bộ kích thước traces với weights ngay sau khi tạo xong tuples
        network.init_traces();

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

        let mut network: NTupleNetwork = Deserialize::deserialize(&mut deserializer)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        // --- BƯỚC QUAN TRỌNG: NÂNG CẤP MODEL CŨ ---
        // 1. Gán giá trị mặc định cho lambda (nếu file cũ không có, serde sẽ dùng default,
        // nhưng ta set lại thủ công cho chắc chắn nếu muốn đổi giá trị khác)
        if network.lambda == 0.0 {
            network.lambda = 0.9;
        }

        // 2. Khởi tạo bộ nhớ cho Traces (vì file cũ không lưu cái này)
        network.init_traces();

        Ok(network)
    }

    // Hàm helper để reset/init traces
    pub fn init_traces(&mut self) {
        self.traces.clear();
        self.active_trace_indices.clear();
        for w_table in &self.weights {
            self.traces.push(vec![0.0; w_table.len()]);
            self.active_trace_indices.push(Vec::with_capacity(1000));
        }
    }

    // Gọi hàm này mỗi khi bắt đầu game mới (trong Env::reset)
    pub fn reset_traces(&mut self) {
        // Cách nhanh: Chỉ zero những thằng đang active
        for (table_idx, indices) in self.active_trace_indices.iter_mut().enumerate() {
            for &feat_idx in indices.iter() {
                self.traces[table_idx][feat_idx] = 0.0;
            }
            indices.clear();
        }
    }

    pub fn update_weights_td_lambda(&mut self, board: &[u32; 16], delta: f64, alpha: f64) {
        // 1. Tính toán Code cho state hiện tại (Giống hàm predict)
        let mut encoded_board = [0usize; 16];
        for i in 0..16 {
            encoded_board[i] = Self::encode_tile(board[i]);
        }

        let decay = self.gamma * self.lambda;
        let threshold = 0.001; // Ngưỡng để cắt đuôi vết (optimization)

        // 2. Bước Decay & Update các vết cũ (Sparse Loop)
        // Duyệt qua tất cả các bảng weights
        for table_idx in 0..self.weights.len() {
            // Chỉ duyệt qua các index đang có vết > 0
            // Dùng retain để vừa duyệt vừa xóa vết nhỏ
            let traces_ref = &mut self.traces[table_idx];
            let weights_ref = &mut self.weights[table_idx];

            self.active_trace_indices[table_idx].retain(|&feat_idx| {
                // Decay
                traces_ref[feat_idx] *= decay;

                // Update Weight: w = w + alpha * delta * e
                weights_ref[feat_idx] += alpha * delta * traces_ref[feat_idx];

                // Giữ lại nếu vết vẫn đủ lớn
                traces_ref[feat_idx].abs() > threshold
            });
        }

        // 3. Bước Cộng thêm vết cho trạng thái HIỆN TẠI (Replacing/Accumulating Traces)
        // Với N-Tuple, ta dùng Accumulating Traces (Cộng dồn)
        for tuple in &self.tuples {
            let table_idx = tuple.weight_index;
            let mut idx = 0;
            for &pos in &tuple.indices {
                idx = idx * 15 + encoded_board[pos];
            }

            // Cộng trace = 1.0 (hoặc += 1.0)
            let old_trace = self.traces[table_idx][idx];
            self.traces[table_idx][idx] += 1.0;

            // Update weight ngay lập tức cho state hiện tại (vì trace vừa tăng)
            // (Lưu ý: Ở bước 2 ta đã update cho trace CŨ rồi, giờ cộng thêm phần delta * 1.0 mới)
            self.weights[table_idx][idx] += alpha * delta * 1.0;

            // Nếu đây là vết mới (từ 0 lên 1), thêm vào danh sách active
            if old_trace.abs() <= threshold {
                self.active_trace_indices[table_idx].push(idx);
            }
        }
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
        let code = ((value as f64 / 3.0).log2() as usize) + 3;
        code.min(14)
    }

    pub fn predict(&self, board: &[u32; 16]) -> f64 {
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

    pub fn predict_game(&self, game: &Game) -> f64 {
        let mut board_flat = [0u32; 16];
        for r in 0..4 {
            for c in 0..4 {
                board_flat[r * 4 + c] = game.board[r][c].value;
            }
        }
        self.predict(&board_flat)
    }

    pub fn update_weights(&mut self, board: &[u32; 16], delta: f64) {
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
