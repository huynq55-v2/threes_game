use std::{
    fs::File,
    io::{BufWriter, Write},
};

use crate::game::Game;

pub struct NTupleNetwork {
    pub tuples: Vec<Vec<usize>>,
    pub weights: Vec<Vec<f32>>,
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
        network.add_rows();
        network.add_cols();
        network.add_squares_2x2();
        network.add_snakes();
        network.add_rectangles_2x3();
        network.add_rectangles_3x2();
        // network.add_rectangles_2x3(); // Disabled as per Java note

        network.init_weights();
        network
    }

    fn add_rows(&mut self) {
        for r in 0..4 {
            self.tuples
                .push(vec![r * 4 + 0, r * 4 + 1, r * 4 + 2, r * 4 + 3]);
        }
    }

    fn add_cols(&mut self) {
        for c in 0..4 {
            self.tuples
                .push(vec![0 * 4 + c, 1 * 4 + c, 2 * 4 + c, 3 * 4 + c]);
        }
    }

    fn add_squares_2x2(&mut self) {
        for r in 0..3 {
            for c in 0..3 {
                self.tuples.push(vec![
                    r * 4 + c,
                    r * 4 + (c + 1),
                    (r + 1) * 4 + c,
                    (r + 1) * 4 + (c + 1),
                ]);
            }
        }
    }

    fn add_snakes(&mut self) {
        self.tuples.push(vec![0, 1, 5, 4]);
        self.tuples.push(vec![3, 2, 6, 7]);
        self.tuples.push(vec![12, 13, 9, 8]);
        self.tuples.push(vec![15, 14, 10, 11]);
    }

    fn add_rectangles_2x3(&mut self) {
        // Các hình chữ nhật nằm ngang (2 hàng, 3 cột)
        for r in 0..3 {
            // r=0,1,2
            for c in 0..2 {
                // c=0,1
                self.tuples.push(vec![
                    r * 4 + c,
                    r * 4 + (c + 1),
                    r * 4 + (c + 2),
                    (r + 1) * 4 + c,
                    (r + 1) * 4 + (c + 1),
                    (r + 1) * 4 + (c + 2),
                ]);
            }
        }
    }

    fn add_rectangles_3x2(&mut self) {
        // Các hình chữ nhật nằm dọc (3 hàng, 2 cột)
        for r in 0..2 {
            // r=0,1
            for c in 0..3 {
                // c=0,1,2
                self.tuples.push(vec![
                    r * 4 + c,
                    r * 4 + (c + 1),
                    (r + 1) * 4 + c,
                    (r + 1) * 4 + (c + 1),
                    (r + 2) * 4 + c,
                    (r + 2) * 4 + (c + 1),
                ]);
            }
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

        // 3. Kiểm tra xem cấu trúc trong code có khớp với file không
        if net.tuples.len() != num_tables {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Mismatch: File has {} tables, but code defines {}",
                    num_tables,
                    net.tuples.len()
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

    fn init_weights(&mut self) {
        for t in &self.tuples {
            let size = 15usize.pow(t.len() as u32);
            self.weights.push(vec![0.0; size]);
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

    pub fn get_index(&self, tuple: &[usize], codes: &[usize; 16]) -> usize {
        let mut index = 0;
        for &pos in tuple {
            index = index * 15 + codes[pos];
        }
        index
    }

    pub fn predict(&self, board_flat: &[u32; 16]) -> f32 {
        let mut codes = [0usize; 16];
        for i in 0..16 {
            codes[i] = Self::encode_tile(board_flat[i]);
        }

        let mut sum = 0.0;
        for (i, tuple) in self.tuples.iter().enumerate() {
            let idx = self.get_index(tuple, &codes);
            sum += self.weights[i][idx];
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

    pub fn update_weights(&mut self, board_flat: &[u32; 16], delta: f32) {
        let mut codes = [0usize; 16];
        for i in 0..16 {
            codes[i] = Self::encode_tile(board_flat[i]);
        }

        for (i, tuple) in self.tuples.iter().enumerate() {
            let idx = self.get_index(tuple, &codes);
            // Cập nhật giá trị vào bảng trọng số
            self.weights[i][idx] += delta;
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
