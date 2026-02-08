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
    pub is_afterstate: bool,
    pub possible_spawn_positions: Vec<usize>,
    pub num_move: u32,
    pub numbers: PseudoList<u32>,
    pub special: PseudoList<u32>,
    pub future_value: u32,
    pub hints: Vec<u32>,
    pub deck_tracker: DeckTracker,
}

impl Game {
    pub fn new() -> Self {
        let is_afterstate = false;
        let possible_spawn_positions = Vec::new();
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
            is_afterstate,
            possible_spawn_positions,
            board,
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

    pub fn set_tile_at_position(&mut self, pos: usize, tile: Tile) {
        let row = pos / 4;
        let col = pos % 4;
        self.board[row][col] = tile;
    }

    pub fn get_highest_tile_rank(&self) -> u8 {
        let mut max_rank = 0;
        for r in 0..4 {
            for c in 0..4 {
                let rank = self.board[r][c].rank();
                if rank > max_rank && rank != 21 && rank != 22 {
                    max_rank = rank;
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

    pub fn count_tiles_with_value(&self, val: u32) -> u8 {
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

    pub fn calculate_score(&mut self) -> f64{
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

    pub fn get_next_value(&mut self) -> u32 {
        
        let is_bonus = if self.num_move > 21 {
            self.special.get_next() == Some(1)
        } else {
            false
        };

        if is_bonus {
            let board_highest_rank = self.get_highest_tile_rank();

            let num = board_highest_rank.saturating_sub(K_SPECIAL_DEMOTION);

            if num < 2 {

            } else {
                if num < 4 {
                    return get_value_from_rank(num);
                } else {
                    let mut rng = rng();
                    let r = rng.random_range(4..num + 1);
                    return get_value_from_rank(r);
                }
            }
        }

        self.numbers.get_next().unwrap()
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
                let val_to_show = get_value_from_rank(clamped_rank + 1);
                hints.push(val_to_show);
            }
        }

        hints.sort();
        hints
    }

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

    pub fn get_valid_moves(&self) -> Vec<Direction> {
        let mut moves = Vec::new();
        if self.can_move(Direction::Up) { moves.push(Direction::Up); }
        if self.can_move(Direction::Down) { moves.push(Direction::Down); }
        if self.can_move(Direction::Left) { moves.push(Direction::Left); }
        if self.can_move(Direction::Right) { moves.push(Direction::Right); }
        moves
    }

    pub fn is_game_over(&mut self) -> bool {
        let valid_moves = self.get_valid_moves();
        valid_moves.is_empty()
    }

    pub fn make_full_move(&mut self, dir: Direction) {
        if !self.can_move(dir) {
            panic!("Invalid move");
        }

        let rot = self.get_rotations_needed(dir);

        // A. Xoay để đưa về hướng Left
        self.rotate_board(rot);

        // B. Xử lý logic Move & Merge
        let (_, moved_rows, _) = self.shift_board_left();

        // C. Spawn gạch mới (nếu có di chuyển)
        self.do_spawn_on_row_ends(&moved_rows);
            
        // Cập nhật trạng thái game
        self.num_move += 1;
        self.future_value = self.get_next_value();
        self.hints = self.predict_future();

        // D. Xoay ngược lại về trạng thái gốc
        self.rotate_board(4 - rot);

        self.is_afterstate = false;
    }

    // generate board after make move but before spawn new number
    pub fn gen_afterstate(&self, dir: Direction) -> Game {
        if !self.can_move(dir) {
            panic!("Invalid move");
        }

        // 1. Clone
        let mut temp_game = self.clone(); 
        
        // 2. Rotate -> Shift -> Rotate Back
        let rot = temp_game.get_rotations_needed(dir);
        temp_game.rotate_board(rot);
        
        let (_, moved_rows, _) = temp_game.shift_board_left();

        for &r in &moved_rows {
            // 1. Tọa độ spawn luôn là cột 3 trong hệ tọa độ "trượt trái"
            let (row_rotated, col_rotated) = (r, 3);
            
            // 2. Map về tọa độ gốc (xoay ngược lại 4 - rot)
            let (orig_row, orig_col) = map_coords(row_rotated, col_rotated, 4 - rot);
            
            // 3. Chuyển sang index phẳng 0-15
            let flat_index = orig_row * 4 + orig_col;
            
            temp_game.possible_spawn_positions.push(flat_index); 
        }
        
        temp_game.rotate_board(4 - rot);

        temp_game.is_afterstate = true;

        temp_game
    }

    // generate all possible outcomes after make move (spawned tile, position)
    // input: an afterstate
    // output: a vector of possible outcomes
    pub fn gen_all_possible_outcomes(&self) -> Vec<Game> {
        if !self.is_afterstate {
            panic!("Cannot generate outcomes from non-afterstate");
        }

        let mut outcomes = Vec::new();
        
        // iterate through possible spawn positions
        for &pos in &self.possible_spawn_positions {
            // iterate through hints
            for &hint in &self.hints {
                let mut possible_game = self.clone();
                possible_game.set_tile_at_position(pos, Tile { value: hint });
                outcomes.push(possible_game);
            }
        }

        outcomes
    }

    // Duyệt từng hàng, xử lý dồn sang trái
    pub fn shift_board_left(&mut self) -> (bool, Vec<usize>, Vec<u8>) {
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

    // Xử lý logic Threes cho 1 hàng duy nhất: Chỉ merge/move cặp đầu tiên tìm thấy
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

    pub fn get_actual_spawn_value(&self) -> u32 {
        // Thêm dấu * ở đầu để chuyển &u32 thành u32
        *self.hints.choose(&mut rng()).unwrap()
    }

    pub fn spawn_at(&mut self, row: usize, col: usize, val: u32) {
        let t = Tile::new(val);
        self.board[row][col] = t;
    }

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
    pub fn get_rotations_needed(&self, dir: Direction) -> u8 {
        match dir {
            Direction::Left => 0,
            Direction::Down => 1,
            Direction::Right => 2,
            Direction::Up => 3,
        }
    }

    // Helper: Xoay board k lần 90 độ
    pub fn rotate_board(&mut self, times: u8) {
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
}

// Helper: Map tọa độ
fn map_coords(row: usize, col: usize, rot: u8) -> (usize, usize) {
    match rot % 4 {
        0 => (row, col),
        1 => (3 - col, row),
        2 => (3 - row, 3 - col),
        3 => (col, 3 - row),
        _ => unreachable!(),
    }
}
