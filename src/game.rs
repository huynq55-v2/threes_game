use crate::deck_tracker::DeckTracker;
use crate::pseudo_list::PseudoList;
use crate::threes_const::*;
use crate::tile::Tile;
use crate::tile::{get_rank_from_value, get_value_from_rank};
use rand::rng;
use rand::seq::IndexedRandom;
use rand::seq::SliceRandom;
use rand::Rng; // Cần thiết cho choose()

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    pub fn from_u8(val: u8) -> Self {
        match val {
            0 => Direction::Up,
            1 => Direction::Down,
            2 => Direction::Left,
            3 => Direction::Right,
            _ => unreachable!(),
        }
    }
}

#[derive(Clone)]
pub struct Game {
    pub board: [[Tile; 4]; 4],
    pub score: f64, // Đã thêm field này
    pub is_afterstate: bool,
    pub possible_spawn_positions: Vec<usize>,
    pub num_move: u32,
    pub numbers: PseudoList<u32>,
    pub special: PseudoList<u32>,
    pub future_value: u32,
    pub hints: Vec<u32>,
    pub deck_tracker: DeckTracker,
    pub predicted_future_distribution: Vec<(u32, f64)>,
}

impl Game {
    pub fn new() -> Self {
        let is_afterstate = false;
        let possible_spawn_positions = Vec::new();
        let num_move = 0;
        let future_value = 0;
        let hints = Vec::new();
        let predicted_future_distribution = Vec::new();
        let score = 0.0;

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

        let mut board = [[Tile::new(0); 4]; 4];

        let mut indices: Vec<usize> = (0..16).collect();
        let mut rng = rng();
        indices.shuffle(&mut rng);

        for &idx in indices.iter().take(K_START_SPAWN_NUMBERS as usize) {
            let row = idx / 4;
            let col = idx % 4;

            if let Some(val) = numbers.get_next() {
                let tile = Tile::new(val as u32);
                board[row][col] = tile;
            }
        }

        let mut game = Game {
            board,
            score,
            is_afterstate,
            possible_spawn_positions,
            num_move,
            numbers,
            special,
            future_value,
            hints,
            deck_tracker: DeckTracker::new(),
            predicted_future_distribution,
        };

        game.future_value = game.get_next_value();
        game.hints = game.predict_future();

        game
    }

    // --- CÁC HÀM GETTER / SETTER CƠ BẢN ---

    pub fn set_tile_at_position(&mut self, pos: usize, tile: Tile) {
        let row = pos / 4;
        let col = pos % 4;
        self.board[row][col] = tile;
    }

    pub fn get_highest_tile_rank(&self) -> u32 {
        let mut max_rank = 0;
        for r in 0..4 {
            for c in 0..4 {
                let rank = self.board[r][c].rank();
                if rank > max_rank && rank != 21 && rank != 22 {
                    max_rank = rank;
                }
            }
        }
        max_rank as u32
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

    pub fn calculate_score(&self) -> f64 {
        let mut total_score = 0;
        for r in 0..4 {
            for c in 0..4 {
                let val = self.board[r][c].value;
                if val >= 3 {
                    let rank = get_rank_from_value(val);
                    total_score += 3_u32.pow(rank as u32);
                }
            }
        }
        total_score as f64
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

    // --- LOGIC GAME & DECK ---

    pub fn get_next_value(&mut self) -> u32 {
        let is_bonus = if self.num_move > 21 {
            self.special.get_next() == Some(1)
        } else {
            false
        };

        if is_bonus {
            let board_highest_rank = self.get_highest_tile_rank();
            let num = board_highest_rank.saturating_sub(K_SPECIAL_DEMOTION);

            if num >= 2 {
                if num < 4 {
                    return get_value_from_rank(num);
                } else {
                    let mut rng = rng();
                    let r = rng.random_range(4..num + 1);
                    return get_value_from_rank(r);
                }
            }
        }
        self.numbers.get_next().unwrap_or(1) // Fallback an toàn
    }

    pub fn predict_future(&self) -> Vec<u32> {
        let mut hints = Vec::new();

        if self.future_value <= 3 {
            hints.push(self.future_value);
        } else {
            let rank = get_rank_from_value(self.future_value);
            let num = (rank.saturating_sub(1)).min(3);

            for i in 0..num {
                let r_idx = rank.saturating_sub(1).saturating_sub(i);
                let clamped_rank = r_idx.clamp(1, 11);
                let val_to_show = get_value_from_rank(clamped_rank as u32 + 1);
                hints.push(val_to_show);
            }
        }
        hints.sort();
        hints
    }

    // --- LOGIC CHECK NƯỚC ĐI (TƯỜNG MINH) ---

    // Helper: Kiểm tra luật gộp của Threes
    fn can_merge_tiles(target: u32, source: u32) -> bool {
        if source == 0 {
            return false;
        } // Không có gì để đẩy
        if target == 0 {
            return true;
        } // Đẩy vào ô trống

        // Gộp 1 + 2 = 3
        if (target == 1 && source == 2) || (target == 2 && source == 1) {
            return true;
        }
        // Gộp X + X (X >= 3)
        if target >= 3 && target == source {
            return true;
        }
        false
    }

    pub fn can_move(&self, dir: Direction) -> bool {
        match dir {
            Direction::Left => {
                for r in 0..4 {
                    for c in 0..3 {
                        if Self::can_merge_tiles(self.board[r][c].value, self.board[r][c + 1].value)
                        {
                            return true;
                        }
                    }
                }
            }
            Direction::Right => {
                for r in 0..4 {
                    for c in (1..4).rev() {
                        // 3, 2, 1
                        if Self::can_merge_tiles(self.board[r][c].value, self.board[r][c - 1].value)
                        {
                            return true;
                        }
                    }
                }
            }
            Direction::Up => {
                for c in 0..4 {
                    for r in 0..3 {
                        if Self::can_merge_tiles(self.board[r][c].value, self.board[r + 1][c].value)
                        {
                            return true;
                        }
                    }
                }
            }
            Direction::Down => {
                for c in 0..4 {
                    for r in (1..4).rev() {
                        // 3, 2, 1
                        if Self::can_merge_tiles(self.board[r][c].value, self.board[r - 1][c].value)
                        {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    pub fn get_valid_moves(&self) -> Vec<Direction> {
        let mut moves = Vec::new();
        if self.can_move(Direction::Up) {
            moves.push(Direction::Up);
        }
        if self.can_move(Direction::Down) {
            moves.push(Direction::Down);
        }
        if self.can_move(Direction::Left) {
            moves.push(Direction::Left);
        }
        if self.can_move(Direction::Right) {
            moves.push(Direction::Right);
        }
        moves
    }

    pub fn is_game_over(&self) -> bool {
        self.get_valid_moves().is_empty()
    }

    // --- LOGIC THỰC HIỆN NƯỚC ĐI (CORE LOGIC) ---

    // Helper: Tính kết quả gộp (Giá trị mới, Điểm thưởng)
    fn get_merge_result(target: u32, source: u32) -> (u32, f64) {
        if target == 0 {
            (source, 0.0)
        } else if target + source == 3 {
            (3, 1.0) // Gộp ra 3 được 1 điểm (hoặc tùy chỉnh)
        } else {
            let new_val = target * 2;
            (new_val, new_val as f64) // Gộp ra X được X điểm
        }
    }

    // Helper: Xử lý logic trượt cho 1 mảng 4 phần tử (theo chiều Left)
    fn process_line_threes(line: [u32; 4]) -> ([u32; 4], bool, f64) {
        let mut new_line = line;
        for i in 0..3 {
            let target = new_line[i];
            let source = new_line[i + 1];

            if Self::can_merge_tiles(target, source) {
                let (new_val, score) = Self::get_merge_result(target, source);
                new_line[i] = new_val;
                // Shift phần còn lại
                for k in (i + 1)..3 {
                    new_line[k] = new_line[k + 1];
                }
                new_line[3] = 0;
                return (new_line, true, score);
            }
        }
        (line, false, 0.0)
    }

    // 1. MOVE & MERGE (Tạo Afterstate)
    // Trả về: (Có di chuyển ko, Danh sách index hàng/cột bị thay đổi)
    pub fn apply_move_no_spawn(&mut self, dir: Direction) -> (bool, Vec<usize>) {
        let mut moved_indices = Vec::new();
        let mut total_score_gain = 0.0;
        let mut has_moved = false;

        match dir {
            Direction::Left => {
                for r in 0..4 {
                    let line = [
                        self.board[r][0].value,
                        self.board[r][1].value,
                        self.board[r][2].value,
                        self.board[r][3].value,
                    ];
                    let (new_line, moved, score) = Self::process_line_threes(line);
                    if moved {
                        for c in 0..4 {
                            self.board[r][c] = Tile::new(new_line[c]);
                        }
                        moved_indices.push(r);
                        total_score_gain += score;
                        has_moved = true;
                    }
                }
            }
            Direction::Right => {
                for r in 0..4 {
                    let line = [
                        self.board[r][3].value,
                        self.board[r][2].value,
                        self.board[r][1].value,
                        self.board[r][0].value,
                    ];
                    let (new_line, moved, score) = Self::process_line_threes(line);
                    if moved {
                        for c in 0..4 {
                            self.board[r][3 - c] = Tile::new(new_line[c]);
                        }
                        moved_indices.push(r);
                        total_score_gain += score;
                        has_moved = true;
                    }
                }
            }
            Direction::Up => {
                for c in 0..4 {
                    let line = [
                        self.board[0][c].value,
                        self.board[1][c].value,
                        self.board[2][c].value,
                        self.board[3][c].value,
                    ];
                    let (new_line, moved, score) = Self::process_line_threes(line);
                    if moved {
                        for r in 0..4 {
                            self.board[r][c] = Tile::new(new_line[r]);
                        }
                        moved_indices.push(c);
                        total_score_gain += score;
                        has_moved = true;
                    }
                }
            }
            Direction::Down => {
                for c in 0..4 {
                    let line = [
                        self.board[3][c].value,
                        self.board[2][c].value,
                        self.board[1][c].value,
                        self.board[0][c].value,
                    ];
                    let (new_line, moved, score) = Self::process_line_threes(line);
                    if moved {
                        for r in 0..4 {
                            self.board[3 - r][c] = Tile::new(new_line[r]);
                        }
                        moved_indices.push(c);
                        total_score_gain += score;
                        has_moved = true;
                    }
                }
            }
        }

        self.score += total_score_gain;
        (has_moved, moved_indices)
    }

    // 2. SPAWN LOGIC
    pub fn spawn_new_tile(&mut self, dir: Direction, moved_indices: &[usize], vals: Vec<u32>) {
        if moved_indices.is_empty() {
            return;
        }

        let mut rng = rng();
        // Chọn ngẫu nhiên 1 hàng/cột trong số những cái đã di chuyển
        let target_index = *moved_indices.choose(&mut rng).unwrap();

        // random val from vals
        let val = *vals.choose(&mut rng).unwrap();

        self.deck_tracker.update(val);

        match dir {
            Direction::Left => self.board[target_index][3] = Tile::new(val),
            Direction::Right => self.board[target_index][0] = Tile::new(val),
            Direction::Up => self.board[3][target_index] = Tile::new(val),
            Direction::Down => self.board[0][target_index] = Tile::new(val),
        }
    }

    // --- CÁC HÀM STATE TRANSITION (HIGH LEVEL) ---

    // Dùng cho Game thật (Environment)
    pub fn make_full_move(&mut self, dir: Direction) {
        if !self.can_move(dir) {
            panic!("Invalid move: {:?}", dir);
        }

        // 1. Move & Merge
        let (has_moved, moved_indices) = self.apply_move_no_spawn(dir);
        if !has_moved {
            panic!("Logic Error: can_move is true but apply_move_no_spawn returned false");
        }

        // 2. Spawn tile in hints
        self.spawn_new_tile(dir, &moved_indices, self.hints.clone());

        // 3. Update State
        self.num_move += 1;
        self.future_value = self.get_next_value();
        self.hints = self.predict_future();
        self.is_afterstate = false;
    }

    // Dùng cho AI Search (Tạo Afterstate để đánh giá)
    pub fn gen_afterstate(&self, dir: Direction) -> Game {
        if !self.can_move(dir) {
            self.print_board();
            panic!("Invalid move: {:?}", dir);
        }

        let mut temp_game = self.clone();

        // 1. Thực hiện Move (chưa spawn)
        let (_, moved_indices) = temp_game.apply_move_no_spawn(dir);

        // 2. Tính toán các vị trí có thể spawn (Để phục vụ Chance Node)
        // Logic tường minh, không map coords phức tạp
        temp_game.possible_spawn_positions.clear();
        for &idx in &moved_indices {
            let flat_pos = match dir {
                Direction::Left => idx * 4 + 3,  // Hàng idx, Cột 3
                Direction::Right => idx * 4 + 0, // Hàng idx, Cột 0
                Direction::Up => 3 * 4 + idx,    // Hàng 3, Cột idx
                Direction::Down => 0 * 4 + idx,  // Hàng 0, Cột idx
            };
            temp_game.possible_spawn_positions.push(flat_pos);
        }

        temp_game.is_afterstate = true;
        temp_game
    }

    // Sinh ra các trạng thái Full State từ Afterstate (Dùng cho Chance Node)
    pub fn gen_all_possible_outcomes(&self) -> Vec<(Game, f64)> {
        if !self.is_afterstate {
            panic!("Cannot generate outcomes from non-afterstate");
        }

        let mut outcomes = Vec::new();

        // P(Vị trí)
        let num_pos = self.possible_spawn_positions.len() as f64;
        let p_pos = 1.0 / num_pos;

        // P(Giá trị) - Tạm tính đều theo Hints hiện tại
        let p_curr_val = 1.0 / self.hints.len() as f64;

        for &pos in &self.possible_spawn_positions {
            for &val in &self.hints {
                // A. Clone ra trạng thái mới
                let mut next_game = self.clone();
                next_game.set_tile_at_position(pos, Tile::new(val));
                next_game.is_afterstate = false;

                // B. Update Tracker & Predict Future Distribution
                next_game.deck_tracker.update(val);
                let max_rank = next_game.get_highest_tile_rank() as u32;
                next_game.predicted_future_distribution =
                    next_game.deck_tracker.predict_future(max_rank);

                // C. Tính xác suất tổng hợp
                let final_prob = p_pos * p_curr_val;

                outcomes.push((next_game, final_prob));
            }
        }

        outcomes
    }

    pub fn print_board(&self) {
        for r in 0..4 {
            for c in 0..4 {
                print!("{:4} ", self.board[r][c].value);
            }
            println!();
        }
    }
}
