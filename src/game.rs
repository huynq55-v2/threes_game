use crate::deck_tracker::DeckTracker;
use crate::pseudo_list::PseudoList;
use crate::threes_const::*;
use crate::tile::Tile;
use crate::tile::{get_rank_from_value, get_value_from_rank};
use rand::Rng;
use rand::rng;
use rand::seq::IndexedRandom;
use rand::seq::SliceRandom;
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

#[derive(Clone)]
pub struct Game {
    pub board: [[Tile; 4]; 4],
    pub score: f64, // calculate at each step
    pub game_over: bool,
    pub num_move: u32,
    pub numbers: PseudoList<u32>,
    pub special: PseudoList<u32>,
    pub future_value: u32,
    pub hints: Vec<u32>,
    pub deck_tracker: DeckTracker,
}

impl Game {
    pub fn new() -> Self {
        let score = 0.0;
        let game_over = false;
        let num_move = 0;
        let future_value = 0;
        let hints = Vec::new();

        let mut numbers = PseudoList::new(K_NUMBER_RANDOMNESS);
        numbers.add(1);
        numbers.add(2);
        numbers.add(3);
        numbers.generate_list();
        numbers.shuffle();

        let mut special = PseudoList::new(1);
        special.add(1);
        for _ in 0..K_SPECIAL_RARENESS {
            special.add(0);
        }
        special.generate_list();
        special.shuffle();

        // Initialize board with empty tiles (0)
        let mut board = [[Tile::new(0); 4]; 4];

        // Create a list of all board positions (0..16)
        let mut indices: Vec<usize> = (0..16).collect();
        // Shuffle to pick random positions
        let mut rng = rng();
        indices.shuffle(&mut rng);

        // Take the first K_START_SPAWN_NUMBERS positions (usually 9)
        for &idx in indices.iter().take(K_START_SPAWN_NUMBERS as usize) {
            let row = idx / 4;
            let col = idx % 4;

            // Get next value from the "deck"
            if let Some(val) = numbers.get_next() {
                let tile = Tile::new(val as u32);
                board[row][col] = tile;
            }
        }

        let mut game = Game {
            board,
            score,
            game_over,
            num_move,
            numbers,
            special,
            future_value,
            hints,
            deck_tracker: DeckTracker::new(),
        };

        // calculate score
        game.calculate_score();

        game.future_value = game.get_next_value();
        game.hints = game.predict_future();

        game
    }

    pub fn get_highest_rank(&self) -> u8 {
        let mut max_rank = 0;
        for r in 0..4 {
            for c in 0..4 {
                let rank = self.board[r][c].rank();
                
                // BỎ QUA các rank đặc biệt 21 (số 1) và 22 (số 2)
                // Chỉ tính những rank thuộc chuỗi nén (3, 6, 12... tương ứng rank 1, 2, 3...)
                if rank != 21 && rank != 22 {
                    if rank > max_rank {
                        max_rank = rank;
                    }
                }
            }
        }
        max_rank
    }
    
    pub fn get_highest_tile_value(&self) -> u32 {
        let mut max_val = 0;
        for r in 0..4 {
            for c in 0..4 {
                let val = self.board[r][c].value;
                if val > max_val {
                    max_val = val;
                }
            }
        }
        max_val
    }

    pub fn count_tiles_with_value(&self, val: u32) -> u32 {
        let mut count = 0;
        for r in 0..4 {
            for c in 0..4 {
                if self.board[r][c].value == val {
                    count += 1;
                }
            }
        }
        count
    }

    pub fn calculate_score(&mut self) {
        let mut total_score = 0;
        for r in 0..4 {
            for c in 0..4 {
                let val = self.board[r][c].value;
                if val >= 3 {
                    let rank = get_rank_from_value(val);
                    // 3^rank
                    total_score += 3_u32.pow(rank as u32);
                }
            }
        }
        self.score = total_score as f64;
    }

    pub fn get_board_flat(&self) -> [u32; 16] {
        let mut flat = [0u32; 16];
        for r in 0..4 {
            for c in 0..4 {
                flat[r * 4 + c] = self.board[r][c].value;
            }
        }
        flat
    }

    pub fn get_next_value(&mut self) -> u32 {
        // C# logic: if (numMoves > 21 && special.GetNext() == 1)
        // Note: special.GetNext() modifies state, so we only call it if numMoves > 21
        let is_bonus = if self.num_move > 21 {
            self.special.get_next() == Some(1)
        } else {
            false
        };

        if is_bonus {
            let board_highest_rank = self.get_highest_rank();
            // int num = Mathf.Max(GetHighestRank() - settings.kSpecialDemotion, 0);
            let num = board_highest_rank.saturating_sub(K_SPECIAL_DEMOTION);

            if num < 2 {
                // Fall back to normal deck
            } else {
                if num < 4 {
                    return get_value_from_rank(num);
                } else {
                    // return GetValue(Random.Range(4, num + 1));
                    // Rust range 4..num+1 excludes num+1, so it covers 4..=num
                    let mut rng = rng();
                    let r = rng.random_range(4..num + 1);
                    return get_value_from_rank(r);
                }
            }
        }

        self.numbers.get_next().unwrap()
    }

    /// Predicts the next tile(s) to show as a hint.
    /// Returns just the hint tiles to be displayed in the UI.
    pub fn predict_future(&self) -> Vec<u32> {
        let mut hints = Vec::new();

        if self.future_value <= 3 {
            hints.push(self.future_value);
        } else {
            // Bonus tile logic
            let rank = get_rank_from_value(self.future_value);
            // int num = Mathf.Min(rank - 1, 3);
            let num = (rank.saturating_sub(1)).min(3);

            for i in 0..num {
                // FaceLayout.FaceData faceData = ...
                let r_idx = rank.saturating_sub(1).saturating_sub(i);
                let clamped_rank = r_idx.clamp(1, 11);
                let val_to_show = get_value_from_rank(clamped_rank + 1);
                hints.push(val_to_show);
            }
        }

        hints.sort();
        hints
    }

    // --- 1. LOGIC KIỂM TRA (READ-ONLY) ---

    pub fn can_move(&self, dir: Direction) -> bool {
        let rot = self.get_rotations_needed(dir);

        for r in 0..4 {
            for c in 0..3 {
                // Ánh xạ tọa độ ảo sang thực
                let (r1, c1) = self.map_rotated_index(r, c, rot);
                let (r2, c2) = self.map_rotated_index(r, c + 1, rot);

                let target = self.board[r1][c1].value;
                let source = self.board[r2][c2].value;

                if source == 0 { continue; } // Không có gạch để đẩy
                
                // Logic merge
                if target == 0 
                   || (target + source == 3) // 1+2 hoặc 2+1
                   || (target >= 3 && target == source) // X+X
                {
                    return true;
                }
            }
        }
        false
    }

    /// Trả về danh sách các hướng đi hợp lệ
    pub fn get_valid_moves(&self) -> Vec<Direction> {
        let mut moves = Vec::new();
        if self.can_move(Direction::Up) { moves.push(Direction::Up); }
        if self.can_move(Direction::Down) { moves.push(Direction::Down); }
        if self.can_move(Direction::Left) { moves.push(Direction::Left); }
        if self.can_move(Direction::Right) { moves.push(Direction::Right); }
        moves
    }

    /// Kiểm tra Game Over
    pub fn check_game_over(&mut self) -> bool {
        let valid_moves = self.get_valid_moves();
        if valid_moves.is_empty() {
            self.game_over = true;
            return true;
        }
        self.game_over = false;
        false
    }

    // --- 2. LOGIC DI CHUYỂN CHÍNH ---

    // Sửa kiểu trả về: (bool, Vec<u8>)
    pub fn move_dir(&mut self, dir: Direction) -> (bool, Vec<u8>) {
        if !self.can_move(dir) {
            return (false, Vec::new());
        }

        let rot = self.get_rotations_needed(dir);

        // A. Xoay để đưa về hướng Left
        self.rotate_board(rot);

        // B. Xử lý logic Move & Merge
        // Lấy danh sách merge từ đây
        let (moved, moved_rows, merged_ranks) = self.shift_board_left();

        // C. Spawn gạch mới (nếu có di chuyển)
        if moved {
            self.do_spawn_on_row_ends(&moved_rows);
            
            // Cập nhật trạng thái game
            self.num_move += 1;
            self.calculate_score();
            self.future_value = self.get_next_value();
            self.hints = self.predict_future();
        }

        // D. Xoay ngược lại về trạng thái gốc
        self.rotate_board(4 - rot);

        (moved, merged_ranks)
    }

    /// Hàm mới: Trả về bàn cờ SAU khi đi, nhưng TRƯỚC khi spawn số mới
    pub fn get_afterstate(&self, dir: Direction) -> Option<[[Tile; 4]; 4]> {
        if !self.can_move(dir) {
            return None;
        }

        // Clone game ra nháp để không ảnh hưởng game thật
        let mut temp_game = self.clone(); 
        
        // Thực hiện logic xoay và dồn gạch (giống move_dir)
        let rot = temp_game.get_rotations_needed(dir);
        temp_game.rotate_board(rot);
        
        let (moved, _, _) = temp_game.shift_board_left();
        
        // Xoay ngược lại
        temp_game.rotate_board(4 - rot);

        if moved {
            Some(temp_game.board)
        } else {
            None
        }
    }

    pub fn get_all_possible_outcomes(&self, dir: Direction) -> Vec<Game> {
        if !self.can_move(dir) {
            return Vec::new();
        }

        let mut outcomes = Vec::new();
        let rot = self.get_rotations_needed(dir);

        // 1. Giả lập cú trượt
        let mut temp_game = self.clone();
        temp_game.rotate_board(rot);
        let (moved, moved_rows, _) = temp_game.shift_board_left();

        if moved {
            // 2. Lấy TẤT CẢ giá trị có thể mọc từ predict_future
            // Thay vì lấy 1 giá trị random, ta lấy list các khả năng
            let possible_spawn_values = temp_game.predict_future();

            // 3. Nhân chéo: Mỗi hàng mọc x Mỗi giá trị khả thi
            for &row_idx in &moved_rows {
                for &val in &possible_spawn_values {
                    let mut possible_game = temp_game.clone();
                    
                    // Spawn cố định vào kịch bản này
                    possible_game.spawn_at(row_idx, 3, val);
                    possible_game.deck_tracker.update(val);

                    // Cập nhật trạng thái như một bước đi thật
                    possible_game.num_move += 1;
                    possible_game.calculate_score();
                    // Quan trọng: Cập nhật future cho bước kế tiếp
                    possible_game.future_value = possible_game.get_next_value();
                    possible_game.hints = possible_game.predict_future();

                    // 4. Xoay ngược lại
                    possible_game.rotate_board(4 - rot);
                    
                    outcomes.push(possible_game);
                }
            }
        }
        outcomes
    }

    pub fn get_all_possible_outcomes_pure(&self, dir: Direction) -> Vec<Game> {
        if !self.can_move(dir) {
            return Vec::new();
        }

        let mut outcomes = Vec::new();
        let rot = self.get_rotations_needed(dir);

        // 1. Giả lập cú trượt (Afterstate sơ khai)
        let mut temp_game = self.clone();
        temp_game.rotate_board(rot);
        let (moved, moved_rows, _) = temp_game.shift_board_left();

        if moved {
            // 2. Lấy danh sách các giá trị có thể mọc từ Future Value
            // Lưu ý: possible_spawn_values sẽ trả về 1 con (nếu là 1,2,3) 
            // hoặc 1-3 con (nếu là Bonus) dựa trên future_value hiện tại
            let possible_spawn_values = self.predict_future();

            // 3. Duyệt qua các hàng có thể mọc gạch và các giá trị khả thi
            for &row_idx in &moved_rows {
                for &val in &possible_spawn_values {
                    let mut possible_game = temp_game.clone();
                    
                    // Đặt gạch vào board (cột 3 là cột mới mọc sau khi trượt trái)
                    possible_game.board[row_idx][3] = Tile { value: val };
                    
                    // QUAN TRỌNG: Cập nhật tracker ngay để các tầng sâu hơn biết con này đã ra
                    possible_game.deck_tracker.update(val);

                    // Xoay ngược lại để về hướng ban đầu
                    possible_game.rotate_board(4 - rot);
                    
                    // Tối giản: Không tính score, không tính hints ở đây để tăng tốc
                    outcomes.push(possible_game);
                }
            }
        }
        outcomes
    }

    // --- 3. CÁC HÀM XỬ LÝ LOGIC CỐT LÕI (CORE LOGIC) ---

    /// Duyệt từng hàng, xử lý dồn sang trái
    fn shift_board_left(&mut self) -> (bool, Vec<usize>, Vec<u8>) {
        let mut moved_rows = Vec::new();
        let mut merged_ranks = Vec::new(); // <--- Danh sách thu hoạch

        for r in 0..4 {
            let (moved, rank_opt) = self.process_single_row(r);
            if moved {
                moved_rows.push(r);
                if let Some(rank) = rank_opt {
                    merged_ranks.push(rank);
                }
            }
        }

        (!moved_rows.is_empty(), moved_rows, merged_ranks)
    }

    /// Xử lý logic Threes cho 1 hàng duy nhất: Chỉ merge/move cặp đầu tiên tìm thấy
    fn process_single_row(&mut self, r: usize) -> (bool, Option<u8>) {
        for c in 0..3 {
            let target_val = self.board[r][c].value;
            let source_val = self.board[r][c + 1].value;

            if source_val == 0 { continue; }

            // Check merge inline cho gọn
            let (new_val, is_merge) = if target_val == 0 {
                (source_val, false) // Chỉ là di chuyển vào ô trống
            } else if target_val + source_val == 3 {
                (3, true) // Merge 1+2
            } else if target_val >= 3 && target_val == source_val {
                (target_val * 2, true) // Merge X+X
            } else {
                continue; // Không merge được cặp này, xét cặp tiếp theo
            };

            // Thực hiện Move & Shift
            self.board[r][c] = Tile::new(new_val);
            
            // Kéo đuôi phía sau lên 1 nấc (Shift Left phần còn lại)
            for k in (c + 1)..3 {
                self.board[r][k] = self.board[r][k + 1];
            }
            self.board[r][3] = Tile::new(0); // Ô cuối luôn trống

            // Tính Rank trả về nếu là Merge
            let merged_rank = if is_merge {
                let r = get_rank_from_value(new_val);
                Some(r)
            } else {
                None
            };

            return (true, merged_rank);
        }
        (false, None)
    }

    // --- 4. CÁC HÀM HELPER & SPAWN ---
    
    // Hàm 1: Quyết định giá trị thực tế sẽ rơi xuống dựa trên future_value (Master Tile)
    pub fn get_actual_spawn_value(&self) -> u32 {
        let mut rng = rng();
        let val = self.future_value;

        if val > 3 {
            let rank = get_rank_from_value(val);
            // Re-roll logic: Lấy trong khoảng [rank-2, rank]
            let min_r = 2.max(rank.saturating_sub(2));
            let actual_rank = rng.random_range(min_r..=rank);
            get_value_from_rank(actual_rank)
        } else {
            val
        }
    }

    // Hàm 2: Thực thi việc đặt quân bài lên board
    pub fn spawn_at(&mut self, row: usize, col: usize, val: u32) {
        let t = Tile::new(val);
        self.board[row][col] = t;
    }

    // Hàm wrapper cũ (đã được làm sạch)
    pub fn do_spawn_on_row_ends(&mut self, moved_rows: &[usize]) {
        if moved_rows.is_empty() { return; }

        let mut rng = rng();
        let target_row = *moved_rows.choose(&mut rng).unwrap();
        
        // Gọi hàm lấy giá trị
        let val = self.get_actual_spawn_value();
        
        // Gọi hàm thực thi (mặc định spawn ở cột cuối - cột 3)
        self.spawn_at(target_row, 3, val);
    }

    // Helper: Lấy số lần xoay cần thiết
    fn get_rotations_needed(&self, dir: Direction) -> u8 {
        match dir {
            Direction::Left => 0,
            Direction::Down => 1,
            Direction::Right => 2,
            Direction::Up => 3,
        }
    }

    // Helper: Xoay board k lần 90 độ
    fn rotate_board(&mut self, times: u8) {
        let k = times % 4;
        if k == 0 { return; }
        
        // Dùng biến tạm để swap. 4x4 stack allocation rất rẻ.
        for _ in 0..k {
            let mut new_board = [[Tile::new(0); 4]; 4];
            for r in 0..4 {
                for c in 0..4 {
                    new_board[c][3 - r] = self.board[r][c];
                }
            }
            self.board = new_board;
        }
    }

    // Helper: Map index (giữ nguyên vì tối ưu)
    fn map_rotated_index(&self, r: usize, c: usize, rot: u8) -> (usize, usize) {
        match rot % 4 {
            0 => (r, c),
            1 => (3 - c, r),
            2 => (3 - r, 3 - c),
            3 => (c, 3 - r),
            _ => unreachable!(),
        }
    }
    
    pub fn get_symmetries(&self) -> Vec<[[Tile; 4]; 4]> {
        let mut symmetries = Vec::with_capacity(8);
        let mut current = self.board;

        // 4 phép xoay của bản gốc
        for _ in 0..4 {
            current = Self::rotate_board_raw(current);
            symmetries.push(current);
        }

        // Lật bản gốc rồi xoay tiếp 4 lần
        let mut flipped = Self::flip_board_x_raw(self.board);
        for _ in 0..4 {
            flipped = Self::rotate_board_raw(flipped);
            symmetries.push(flipped);
        }

        symmetries
    }

    fn rotate_board_raw(board: [[Tile; 4]; 4]) -> [[Tile; 4]; 4] {
        let mut new_board = [[Tile::new(0); 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                new_board[c][3 - r] = board[r][c];
            }
        }
        new_board
    }

    fn flip_board_x_raw(board: [[Tile; 4]; 4]) -> [[Tile; 4]; 4] {
        let mut new_board = [[Tile::new(0); 4]; 4];
        for r in 0..4 {
            for c in 0..4 {
                new_board[r][3 - c] = board[r][c];
            }
        }
        new_board
    }
}

impl Game {
    /// Creates a new Game with a specific board state for testing.
    #[allow(dead_code)] // Useful for integration tests even if not used in main binary yet
    pub fn new_with_board(board: [[Tile; 4]; 4], num_move: u32) -> Self {
        let score = 0.0;
        let game_over = false;
        let future_value = 0;
        let hints = Vec::new();

        // Standard initialization for lists
        let mut numbers = PseudoList::new(K_NUMBER_RANDOMNESS);
        numbers.add(1);
        numbers.add(2);
        numbers.add(3);
        numbers.generate_list();
        numbers.shuffle();

        let mut special = PseudoList::new(1);
        special.add(1);
        // for _ in 0..K_SPECIAL_RARENESS {
        //     special.add(0);
        // }
        special.generate_list();
        special.shuffle();

        let mut game = Game {
            board,
            score,
            game_over,
            num_move,
            numbers,
            special,
            future_value,
            hints,
            deck_tracker: DeckTracker::new(),
        };

        game.future_value = game.get_next_value();
        game.hints = game.predict_future();

        game
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_init() {
        let game = Game::new();

        // Count non-empty tiles
        let mut count = 0;
        let mut ones = 0;
        let mut twos = 0;
        let mut threes = 0;

        for r in 0..4 {
            for c in 0..4 {
                let val = game.board[r][c].value;
                if val != 0 {
                    count += 1;
                    if val == 1 {
                        ones += 1;
                    } else if val == 2 {
                        twos += 1;
                    } else if val == 3 {
                        threes += 1;
                    }
                }
            }
        }

        assert_eq!(count, K_START_SPAWN_NUMBERS as u32);

        // Check fairness/distribution if possible, but randomness makes exact counts vary.
        // However, with multiplier 4, we have (4,4,4) of each in bag.
        // We pick 9.
        // Max "fluctuation" isn't strictly defined without calculating probability,
        // but we know we can't have 5 ones, etc.
        // But simply checking we have 9 tiles of values 1,2,3 is good enough for now.
        assert!(ones + twos + threes == count);
    }

    #[test]
    fn test_predict_future_with_rank_8() {
        // 1. Setup board with one tile of Rank 8 (Value 384)
        // Value 384: (384/3).ilog2() + 1 = 128.ilog2() + 1 = 7 + 1 = 8.
        let mut board = [[Tile::new(0); 4]; 4];
        board[0][0] = Tile::new(384);

        // 2. Set condition for Bonus
        // Rule: num_move > 21
        let num_move = 22;

        let game = Game::new_with_board(board, num_move);

        // 4. Verify results
        let valid_a = vec![12, 24, 48];
        let valid_b = vec![6, 12, 24];

        assert!(
            game.hints == valid_a || game.hints == valid_b,
            "Hints {:?} are not valid. Expected A: {:?} or B: {:?}",
            game.hints,
            valid_a,
            valid_b
        );
    }

    #[test]
    fn test_bug_report_down_move() {
        // 0 0 2 0
        // 0 2 0 3
        // 0 0 1 1
        // 3 6 3 1
        let mut board = [[Tile::new(0); 4]; 4];
        board[0] = [Tile::new(0), Tile::new(0), Tile::new(2), Tile::new(0)];
        board[1] = [Tile::new(0), Tile::new(2), Tile::new(0), Tile::new(3)];
        board[2] = [Tile::new(0), Tile::new(0), Tile::new(1), Tile::new(1)];
        board[3] = [Tile::new(3), Tile::new(6), Tile::new(3), Tile::new(1)];
        
        let game = Game::new_with_board(board, 10);
        let can = game.can_move(Direction::Down);
        assert!(can, "Should be able to move Down");
    }

    #[test]
    fn test_spawn_value_always_consistent_with_hints() {
        let mut rng = rng();

        // Chạy 100 kịch bản bàn cờ khác nhau
        for _ in 0..100 {
            // 1. Random một Rank từ 1 đến 12 để đặt lên bàn
            let random_rank = rng.random_range(1..=12);
            let tile_value = get_value_from_rank(random_rank);
            
            let mut board = [[Tile::new(0); 4]; 4];
            board[0][0] = Tile::new(tile_value);

            // 2. Tạo game (num_move > 21 để kích hoạt khả năng bốc Bonus)
            // future_value sẽ tự động được tính toán bên trong new_with_board
            let mut game = Game::new_with_board(board, 25);

            // 3. Lấy danh sách Hint mà AI nhìn thấy
            let hints = game.predict_future();
            
            // 4. Kiểm tra 100 lần spawn thực tế cho mỗi kịch bản
            for _ in 0..100 {
                let actual_val = game.get_actual_spawn_value();
                
                // Assert: Giá trị thực tế RƠI xuống PHẢI nằm trong danh sách ĐÃ HỨA (Hints)
                assert!(
                    hints.contains(&actual_val),
                    "\n[LỖI MISMATCH]\n\
                     Board Max Rank: {} (Value: {})\n\
                     Future Master Value (Hidden): {}\n\
                     Hints Displayed: {:?}\n\
                     Actual Spawned: {}\n",
                    random_rank, tile_value, game.future_value, hints, actual_val
                );
            }
        }
        println!("✅ Đã kiểm tra 10.000 lần spawn, tất cả đều khớp với Hint!");
    }

    #[test]
    fn test_symmetries_uniqueness() {
        // Tạo bàn cờ test với các giá trị khác biệt hoàn toàn
        // 0  1  2  3
        // 4  5  6  7
        // 8  9 10 11
        // 12 13 14 15
        let mut game = Game::new(); 
        for i in 0..16 {
            game.board[i / 4][i % 4].value = i as u32;
        }

        let syms = game.get_symmetries();
        
        // 1. Kiểm tra số lượng phải là 8
        assert_eq!(syms.len(), 8);

        // 2. Kiểm tra tính độc nhất (không ma trận nào trùng nhau)
        for i in 0..8 {
            for j in (i + 1)..8 {
                assert_ne!(syms[i], syms[j], "Symmetry {} and {} are identical!", i, j);
            }
        }
    }

    #[test]
    fn test_specific_symmetry_rotation() {
        let mut game = Game::new();
        game.board[0][0].value = 99; // Chỉ đặt 1 ô ở góc trên trái

        let syms = game.get_symmetries();

        // Trong 8 biến thể của 1 ô góc, nó phải nằm ở các vị trí góc
        for s in syms {
            let val_at_corners = s[0][0].value == 99 || s[0][3].value == 99 || 
                                 s[3][0].value == 99 || s[3][3].value == 99;
            assert!(val_at_corners, "A corner tile must remain in a corner after symmetry");
        }
    }
}
