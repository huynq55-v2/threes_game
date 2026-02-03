use std::{
    fs::File,
    io::{BufWriter, Write},
};

use crate::game::Game;

pub struct TupleConfig {
    pub indices: Vec<usize>, // Các ô trên bàn cờ (ví dụ: [0,1,2,3,7])
    pub weight_index: usize, // Trỏ đến bảng weights số mấy (ví dụ: bảng số 0)
}

pub struct NTupleNetwork {
    pub tuples: Vec<TupleConfig>, // Danh sách 96 con rắn
    pub weights: Vec<Vec<f32>>,   // Chỉ có 12 bảng dữ liệu thôi
    pub alpha: f32,
    pub gamma: f32,
}

impl NTupleNetwork {
    pub fn new(alpha: f32, gamma: f32) -> Self {
        let mut network = NTupleNetwork {
            tuples: Vec::new(),
            weights: Vec::new(),
            alpha,
            gamma,
        };

        network.add_shared_snake();

        // network.init_weights();
        network
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

    // Hàm load để bác dùng sau này
    pub fn load_from_binary(path: &str, alpha: f32, gamma: f32) -> std::io::Result<Self> {
        use std::io::{BufReader, Read};
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // 1. Đọc số lượng tables từ file
        let mut head = [0u8; 4];
        reader.read_exact(&mut head)?;
        let num_tables = u32::from_le_bytes(head) as usize;

        // 2. Khởi tạo một mạng mới với cấu trúc Tuples hiện tại trong code
        let mut net = Self::new(alpha, gamma);

        // Code mới (ĐÚNG):
        if net.weights.len() != num_tables {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Mismatch: File has {} tables, but code expects {} weight tables",
                    num_tables,
                    net.weights.len() // So sánh với số lượng bảng weights thực tế
                ),
            ));
        }

        // 4. Đọc dữ liệu Weights và nạp vào net.weights
        net.weights.clear();
        for _ in 0..num_tables {
            reader.read_exact(&mut head)?;
            let table_size = u32::from_le_bytes(head) as usize;

            let mut table = vec![0.0f32; table_size];
            // Đọc toàn bộ bảng float dưới dạng byte rồi chuyển đổi
            let mut float_buf = vec![0u8; table_size * 4];
            reader.read_exact(&mut float_buf)?;

            for i in 0..table_size {
                let bytes = [
                    float_buf[i * 4],
                    float_buf[i * 4 + 1],
                    float_buf[i * 4 + 2],
                    float_buf[i * 4 + 3],
                ];
                table[i] = f32::from_le_bytes(bytes);
            }
            net.weights.push(table);
        }

        Ok(net)
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

    pub fn export_to_binary(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // 1. Ghi số lượng bảng trọng số (Tuples count)
        writer.write_all(&(self.weights.len() as u32).to_le_bytes())?;

        // 2. Ghi dữ liệu từng bảng
        for table in &self.weights {
            // Ghi kích thước bảng
            writer.write_all(&(table.len() as u32).to_le_bytes())?;
            // Ghi từng giá trị float 32-bit (Little Endian khớp với Java/Android)
            for &weight in table {
                writer.write_all(&weight.to_le_bytes())?;
            }
        }
        writer.flush()?;
        Ok(())
    }
}
