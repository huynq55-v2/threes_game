use crate::game::{Direction, Game};
use crate::ui::{ConsoleUI, InputEvent};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyclass]
pub struct ThreesEnv {
    game: Game,
    history: Vec<Game>,
    future_history: Vec<Game>,
}

#[pymethods]
impl ThreesEnv {
    #[new]
    fn new() -> Self {
        ThreesEnv {
            game: Game::new(),
            history: Vec::new(),
            future_history: Vec::new(),
        }
    }

    fn reset(&mut self) -> Vec<u32> {
        self.game = Game::new();
        self.history.clear();
        self.future_history.clear();
        self.get_board_flat()
    }

    fn step(&mut self, action: u32) -> (Vec<u32>, f32, bool) {
        // Save history before move/checking invalid
        // Actually, only save if move is valid or we just checkpoint everything?
        // Since we return reward, invalid moves are "transitions" too (state doesn't change).
        // If we want exact undo, we should save for every step?
        // Or only save when board changes?
        // User wants "undo", implying reverting moves.

        let dir = match action {
            0 => Direction::Up,
            1 => Direction::Down,
            2 => Direction::Left,
            3 => Direction::Right,
            _ => return (self.get_board_flat(), -10.0, true),
        };

        let old_score = self.game.score;

        // Check valid move first to decide on history?
        if self.game.can_move(dir) {
            self.history.push(self.game.clone());
            self.future_history.clear();
        }

        let moved = self.game.move_dir(dir);
        let new_score = self.game.score;
        let game_over = self.game.check_game_over();

        let reward = if moved {
            let delta = new_score as f32 - old_score as f32;
            if delta > 0.0 {
                (delta + 1.0).log2()
            } else {
                0.1
            }
        } else {
            // invalid move
            -1.0
        };

        let final_reward = if game_over { reward - 10.0 } else { reward };

        (self.get_board_flat(), final_reward, game_over)
    }

    fn valid_moves(&self) -> Vec<bool> {
        let mut mask = vec![false; 4];
        if self.game.can_move(Direction::Up) {
            mask[0] = true;
        }
        if self.game.can_move(Direction::Down) {
            mask[1] = true;
        }
        if self.game.can_move(Direction::Left) {
            mask[2] = true;
        }
        if self.game.can_move(Direction::Right) {
            mask[3] = true;
        }
        mask
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

    fn undo(&mut self) -> PyResult<Vec<u32>> {
        if let Some(prev) = self.history.pop() {
            self.future_history.push(self.game.clone());
            self.game = prev;
        }
        Ok(self.get_board_flat())
    }

    fn redo(&mut self) -> PyResult<Vec<u32>> {
        if let Some(next) = self.future_history.pop() {
            self.history.push(self.game.clone());
            self.game = next;
        }
        Ok(self.get_board_flat())
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
                let moved = game.move_dir(dir);
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
