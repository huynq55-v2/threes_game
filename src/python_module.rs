use crate::game::{Direction, Game};
use crate::ui::{ConsoleUI, InputEvent};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
pub struct ThreesEnv {
    game: Game,
    history: Vec<Game>,
    future_history: Vec<Game>,
    // For reward calculation
    prev_max_val: u32,
    prev_max_count: u32,
    prev_pre_max_count: u32,
    prev_hub_count: u32,
    gamma: f32,
}

#[pymethods]
impl ThreesEnv {
    #[new]
    fn new() -> Self {
        let game = Game::new();
        // Initialize stats from the new game state
        let stats = game.get_game_stats();
        ThreesEnv {
            game,
            history: Vec::new(),
            future_history: Vec::new(),
            prev_max_val: stats.max_val,
            prev_max_count: stats.max_count,
            prev_pre_max_count: stats.pre_max_count,
            prev_hub_count: stats.hub_count,
            gamma: 0.99,
        }
    }

    pub fn set_gamma(&mut self, gamma: f32) {
        self.gamma = gamma;
    }

    // reset return board and hints
    fn reset(&mut self) -> (Vec<u32>, Vec<u32>) {
        // Create new game but preserve the "brain" (RarityEngine)
        // We clone the existing engine to pass into the new game
        self.game = Game::new_with_rarity(self.game.rarity.clone());
        self.history.clear();
        self.future_history.clear();

        let stats = self.game.get_game_stats();
        self.prev_max_val = stats.max_val;
        self.prev_max_count = stats.max_count;
        self.prev_pre_max_count = stats.pre_max_count;
        self.prev_hub_count = stats.hub_count;

        (self.get_board_flat(), self.game.hints.clone())
    }

    // Return signature: (Board, Reward, Done, Hints)
    fn step(&mut self, action: u32) -> (Vec<u32>, f32, bool, Vec<u32>) {
        // 1. Map Action
        let dir = match action {
            0 => Direction::Up,
            1 => Direction::Down,
            2 => Direction::Left,
            3 => Direction::Right,
            _ => return (self.get_board_flat(), -10.0, true, self.game.hints.clone()),
        };

        let phi_old = self.game.calculate_potential_hybrid();

        // 3. Thực hiện Move
        if self.game.can_move(dir) {
            // Lưu history cho Undo nếu cần
            self.history.push(self.game.clone());
            self.future_history.clear();

            let (_, _) = self.game.move_dir(dir);
            let game_over = self.game.check_game_over();

            let phi_new = self.game.calculate_potential_hybrid();

            let shaping_reward = (self.gamma * phi_new) - phi_old;

            (
                self.get_board_flat(),
                shaping_reward,
                game_over,
                self.game.hints.clone(),
            )
        } else {
            // Invalid move logic (Maskable PPO will mask this usually, but handled here just in case)
            unreachable!()
        }
    }

    // Cập nhật hàm Undo để trả về bộ tuple đầy đủ
    fn undo(&mut self) -> PyResult<(Vec<u32>, f32, bool, Vec<u32>)> {
        if let Some(prev) = self.history.pop() {
            // Lưu trạng thái hiện tại vào tương lai trước khi quay lại
            self.future_history.push(self.game.clone());
            self.game = prev;
        }

        // Trả về dữ liệu trạng thái sau khi đã lùi lại
        Ok((
            self.get_board_flat(),
            0.0, // Reward của undo thường là 0
            self.game.check_game_over(),
            self.game.hints.clone(),
        ))
    }

    fn redo(&mut self) -> PyResult<(Vec<u32>, f32, bool, Vec<u32>)> {
        if let Some(next) = self.future_history.pop() {
            self.history.push(self.game.clone());
            self.game = next;
        }

        Ok((
            self.get_board_flat(),
            0.0,
            self.game.check_game_over(),
            self.game.hints.clone(),
        ))
    }

    fn set_board(&mut self, flat_board: Vec<u32>, num_move: u32, next_value: u32) {
        let mut new_board = [[crate::tile::Tile::new(0); 4]; 4];
        for i in 0..16 {
            let r = i / 4;
            let c = i % 4;
            new_board[r][c] = crate::tile::Tile::new(flat_board[i]);
        }

        // Cập nhật trạng thái game
        self.game.board = new_board;
        self.game.num_move = num_move;
        self.game.future_value = next_value;
        self.game.hints = self.game.predict_future(); // Tự tính lại hint
        self.game.calculate_score();

        // Xóa lịch sử cũ để tránh lỗi Undo/Redo
        self.history.clear();
        self.future_history.clear();
    }

    fn valid_moves(&self) -> Vec<bool> {
        vec![
            self.game.can_move(Direction::Up),
            self.game.can_move(Direction::Down),
            self.game.can_move(Direction::Left),
            self.game.can_move(Direction::Right),
        ]
    }

    fn get_hint_set(&self) -> Vec<u32> {
        self.game.hints.clone()
    }

    fn get_symmetries(&self) -> Vec<(Vec<u32>, Vec<u32>)> {
        let boards = self.game.get_symmetries();
        let mut result = Vec::with_capacity(8);

        // 0: Up, 1: Down, 2: Left, 3: Right
        let p_id = vec![0, 1, 2, 3];
        // Rot90: Up->Right(3), Down->Left(2), Left->Up(0), Right->Down(1)
        let p_rot90 = vec![3, 2, 0, 1];
        // Rot180: Up->Down(1), Down->Up(0), Left->Right(3), Right->Left(2)
        let p_rot180 = vec![1, 0, 3, 2];
        // Rot270: Up->Left(2), Down->Right(3), Left->Down(1), Right->Up(0)
        let p_rot270 = vec![2, 3, 1, 0];
        // FlipX: Left<->Right (2<->3)
        let p_flip = vec![0, 1, 3, 2];

        let apply_map = |base: &Vec<u32>, map: &Vec<u32>| -> Vec<u32> {
            base.iter().map(|&x| map[x as usize]).collect()
        };

        let p_flip_r0 = p_flip.clone();
        let p_flip_r90 = apply_map(&p_rot90, &p_flip);
        let p_flip_r180 = apply_map(&p_rot180, &p_flip);
        let p_flip_r270 = apply_map(&p_rot270, &p_flip);

        let perms = vec![
            p_id,
            p_rot90,
            p_rot180,
            p_rot270,
            p_flip_r0,
            p_flip_r90,
            p_flip_r180,
            p_flip_r270,
        ];

        for (i, board) in boards.iter().enumerate() {
            let mut flat = Vec::with_capacity(16);
            for r in 0..4 {
                for c in 0..4 {
                    flat.push(board[r][c].value);
                }
            }
            result.push((flat, perms[i].clone()));
        }

        result
    }

    fn get_board_flat(&self) -> Vec<u32> {
        let mut flat = Vec::with_capacity(16);
        for r in 0..4 {
            for c in 0..4 {
                flat.push(self.game.board[r][c].value);
            }
        }
        flat
    }

    #[getter]
    fn score(&self) -> u32 {
        self.game.score
    }

    // --- UI Integration ---

    fn init_ui(&self) -> PyResult<()> {
        ConsoleUI::init().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    fn cleanup_ui(&self) -> PyResult<()> {
        ConsoleUI::cleanup().map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    fn render_cli(&self, message: Option<String>) -> PyResult<()> {
        let mut valid = Vec::new();
        if self.game.can_move(Direction::Up) {
            valid.push(Direction::Up);
        }
        if self.game.can_move(Direction::Down) {
            valid.push(Direction::Down);
        }
        if self.game.can_move(Direction::Left) {
            valid.push(Direction::Left);
        }
        if self.game.can_move(Direction::Right) {
            valid.push(Direction::Right);
        }

        ConsoleUI::print_game_state(&self.game, &valid, message.as_deref())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }
    fn get_input(&self) -> PyResult<Option<u32>> {
        match ConsoleUI::get_input() {
            Ok(InputEvent::Dir(Direction::Up)) => Ok(Some(0)),
            Ok(InputEvent::Dir(Direction::Down)) => Ok(Some(1)),
            Ok(InputEvent::Dir(Direction::Left)) => Ok(Some(2)),
            Ok(InputEvent::Dir(Direction::Right)) => Ok(Some(3)),
            Ok(InputEvent::Undo) => Ok(Some(10)), // 10 = Undo
            Ok(InputEvent::Redo) => Ok(Some(11)), // 11 = Redo
            Ok(InputEvent::Quit) => Ok(None),
            Err(e) => Err(pyo3::exceptions::PyIOError::new_err(e.to_string())),
        }
    }
}

#[pyclass]
pub struct ThreesVecEnv {
    envs: Vec<Game>,
}

#[pymethods]
impl ThreesVecEnv {
    #[new]
    fn new(num_envs: usize) -> Self {
        let envs = (0..num_envs).map(|_| Game::new()).collect();
        ThreesVecEnv { envs }
    }

    fn reset(&mut self) -> Vec<Vec<u32>> {
        self.envs.par_iter_mut().for_each(|g| *g = Game::new());
        self.envs
            .iter()
            .map(|g| {
                let mut flat = Vec::with_capacity(16);
                for r in 0..4 {
                    for c in 0..4 {
                        flat.push(g.board[r][c].value);
                    }
                }
                flat
            })
            .collect()
    }

    /// Parallel Step
    /// Returns: (boards_batch, rewards_batch, dones_batch)
    fn step(&mut self, actions: Vec<u32>) -> (Vec<Vec<u32>>, Vec<f32>, Vec<bool>) {
        // Parallel execution using Rayon
        // map collect is efficient
        let results: Vec<(Vec<u32>, f32, bool)> = self
            .envs
            .par_iter_mut()
            .zip(actions.par_iter())
            .map(|(game, &action)| {
                let dir = match action {
                    0 => Direction::Up,
                    1 => Direction::Down,
                    2 => Direction::Left,
                    3 => Direction::Right,
                    _ => Direction::Up, // Should handle invalid action gracefully or panic
                };

                let old_score = game.score;
                // Fix signature mismatch
                let (moved, _merged_ranks) = game.move_dir(dir);
                let new_score = game.score;
                let game_over = game.check_game_over();

                let reward = if moved {
                    let delta = new_score as f32 - old_score as f32;
                    if delta > 0.0 {
                        (delta + 1.0).log2()
                    } else {
                        0.05
                    }
                } else {
                    -1.0
                };

                let final_reward = if game_over { reward - 50.0 } else { reward };

                if game_over {
                    // Auto-reset if needed or let Python handle?
                    // Usually VecEnvs auto-reset.
                    *game = Game::new();
                    // Note: returned board is the NEW board (after reset).
                    // In Gym implementation, we usually return `info` with terminal observation.
                    // For simplicity, we just return the new board.
                }

                let mut flat = Vec::with_capacity(16);
                for r in 0..4 {
                    for c in 0..4 {
                        flat.push(game.board[r][c].value);
                    }
                }

                (flat, final_reward, game_over)
            })
            .collect();

        // Unzip manually or 3 iterators?
        let mut boards = Vec::with_capacity(results.len());
        let mut rewards = Vec::with_capacity(results.len());
        let mut dones = Vec::with_capacity(results.len());

        for (b, r, d) in results {
            boards.push(b);
            rewards.push(r);
            dones.push(d);
        }

        (boards, rewards, dones)
    }

    fn get_hint_sets(&self) -> Vec<Vec<u32>> {
        self.envs.iter().map(|g| g.hints.clone()).collect()
    }

    fn valid_moves_batch(&self) -> Vec<Vec<bool>> {
        self.envs
            .par_iter()
            .map(|g| {
                let mut mask = vec![false; 4];
                if g.can_move(Direction::Up) {
                    mask[0] = true;
                }
                if g.can_move(Direction::Down) {
                    mask[1] = true;
                }
                if g.can_move(Direction::Left) {
                    mask[2] = true;
                }
                if g.can_move(Direction::Right) {
                    mask[3] = true;
                }
                mask
            })
            .collect()
    }
}
