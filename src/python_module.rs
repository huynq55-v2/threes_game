use crate::adaptive_manager::AdaptiveManager;
use crate::game::{Direction, Game};
use crate::n_tuple_network::NTupleNetwork;
use crate::pbt::TrainingConfig;
use crate::potential_calculate::get_composite_potential;
use crate::ui::{ConsoleUI, InputEvent};
use pyo3::prelude::*;
use rand::Rng as _;
use rayon::prelude::*;

pub struct ThreesEnv {
    pub game: Game,
    pub history: Vec<Game>,
    pub future_history: Vec<Game>,
    // For reward calculation
    pub prev_max_val: u32,
    pub prev_max_count: u32,
    prev_pre_max_count: u32,
    prev_hub_count: u32,
    gamma: f64,
    adaptive_manager: AdaptiveManager,
    pub config: TrainingConfig,
}

impl ThreesEnv {
    pub fn new(gamma: f64) -> Self {
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
            gamma: gamma,
            adaptive_manager: AdaptiveManager::new(),
            config: TrainingConfig::default(),
        }
    }

    pub fn set_config(&mut self, new_cfg: TrainingConfig) {
        self.config = new_cfg;
    }

    pub fn set_gamma(&mut self, gamma: f64) {
        self.gamma = gamma;
    }

    // reset return board and hints
    pub fn reset(&mut self) -> (Vec<u32>, Vec<u32>) {
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

        (self.get_board_flat().to_vec(), self.game.hints.clone())
    }

    // Return signature: (Board, Reward, Done, Hints)
    fn step(&mut self, action: u32) -> (Vec<u32>, f64, bool, Vec<u32>) {
        let dir = match action {
            0 => Direction::Up,
            1 => Direction::Down,
            2 => Direction::Left,
            3 => Direction::Right,
            _ => {
                return (
                    self.get_board_flat().to_vec(),
                    -10.0,
                    true,
                    self.game.hints.clone(),
                )
            }
        };

        // 1. Lấy dữ liệu TRƯỚC khi move
        let phi_old = get_composite_potential(&self.game.board, &self.config);
        let score_old = self.game.score; // <-- Lưu điểm cũ lại

        if self.game.can_move(dir) {
            self.history.push(self.game.clone());
            self.future_history.clear();

            // 2. Thực hiện Move
            // Lưu ý: biến đầu tiên là 'moved' (bool), ko phải điểm
            let (_, _) = self.game.move_dir(dir);

            let game_over = self.game.check_game_over();

            // 3. Lấy dữ liệu SAU khi move
            let phi_new = get_composite_potential(&self.game.board, &self.config);
            let score_new = self.game.score; // <-- Điểm mới (đã được update trong move_dir)

            let base_reward = (score_new - score_old) as f64;
            let gamma = self.gamma;
            let raw_shaping = (gamma * phi_new) - phi_old;

            let final_reward = self
                .adaptive_manager
                .update_and_scale(base_reward, raw_shaping);

            (
                self.get_board_flat().to_vec(),
                final_reward,
                game_over,
                self.game.hints.clone(),
            )
        } else {
            unreachable!()
        }
    }

    // Cập nhật hàm Undo để trả về bộ tuple đầy đủ
    fn undo(&mut self) -> PyResult<(Vec<u32>, f64, bool, Vec<u32>)> {
        if let Some(prev) = self.history.pop() {
            // Lưu trạng thái hiện tại vào tương lai trước khi quay lại
            self.future_history.push(self.game.clone());
            self.game = prev;
        }

        // Trả về dữ liệu trạng thái sau khi đã lùi lại
        Ok((
            self.get_board_flat().to_vec(),
            0.0, // Reward của undo thường là 0
            self.game.check_game_over(),
            self.game.hints.clone(),
        ))
    }

    fn redo(&mut self) -> PyResult<(Vec<u32>, f64, bool, Vec<u32>)> {
        if let Some(next) = self.future_history.pop() {
            self.history.push(self.game.clone());
            self.game = next;
        }

        Ok((
            self.get_board_flat().to_vec(),
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

    pub fn get_random_valid_action(&self) -> u32 {
        let mut rng = rand::rng();
        let mut action = rng.random_range(0..4);
        while !self.valid_moves()[action] {
            action = rng.random_range(0..4);
        }
        action as u32
    }

    pub fn get_best_action_safe(&self, brain: &NTupleNetwork) -> u32 {
        let mut best_action = 0;
        let mut max_of_min_quality = f64::NEG_INFINITY; // Dùng âm vô cùng cho chuẩn

        let current_board_score = self.game.score;
        let gamma = self.gamma;

        for action_idx in 0..4 {
            let dir = match action_idx {
                0 => Direction::Up,
                1 => Direction::Down,
                2 => Direction::Left,
                3 => Direction::Right,
                _ => continue,
            };

            let outcomes = self.game.get_all_possible_outcomes(dir);

            if !outcomes.is_empty() {
                let mut min_quality = f64::INFINITY;

                for outcome in &outcomes {
                    let score_gain = (outcome.score - current_board_score) as f64;

                    // CHỈ DÙNG PREDICT, bỏ Potential nếu bác đang chơi "Vô Ngã"
                    let future_value = brain.predict_game(outcome);
                    let move_quality = score_gain + gamma * future_value;

                    if move_quality < min_quality {
                        min_quality = move_quality;
                    }
                }

                if min_quality > max_of_min_quality {
                    max_of_min_quality = min_quality;
                    best_action = action_idx;
                }
            }
        }

        // Nếu không tìm được nước đi nào (kẹt cứng), trả về một hành động mặc định
        // hoặc xử lý logic Game Over ở bên ngoài
        best_action as u32
    }

    pub fn get_best_action_expectimax(&self, brain: &NTupleNetwork) -> u32 {
        let mut best_val = -f64::MAX;
        let mut best_action = 0;

        let current_board_score = self.game.score;

        for action_idx in 0..4 {
            let dir = match action_idx {
                0 => Direction::Up,
                1 => Direction::Down,
                2 => Direction::Left,
                3 => Direction::Right,
                _ => continue,
            };

            let outcomes = self.game.get_all_possible_outcomes(dir);

            if !outcomes.is_empty() {
                let mut expected_value = 0.0;
                for outcome in &outcomes {
                    let outcome_score = outcome.score;
                    let score_gain = (outcome_score - current_board_score) as f64;
                    let gamma = self.gamma;
                    let future_value = brain.predict_game(outcome)
                        + get_composite_potential(&outcome.board, &self.config);
                    let move_quality = score_gain + gamma * future_value;

                    expected_value += move_quality;
                }
                expected_value /= outcomes.len() as f64;

                if expected_value > best_val {
                    best_val = expected_value;
                    best_action = action_idx;
                }
            }
        }
        best_action
    }

    pub fn get_best_action_greedy(&mut self, brain: &NTupleNetwork) -> u32 {
        let mut best_action = 0; // Mặc định
        let mut best_value = -f64::MAX;
        let valid_actions = self.game.get_valid_moves();

        if valid_actions.is_empty() {
            return 0; // Hoặc trả về random nếu muốn
        }

        // Duyệt qua các nước đi hợp lệ
        for &action_dir in &valid_actions {
            // Clone game ra để thử nước đi
            let mut virtual_game = self.game.clone();
            virtual_game.move_dir(action_dir); // Đi thử

            // Lấy board phẳng sau khi đi
            let board_flat = virtual_game.get_board_flat();
            // Dự đoán giá trị bằng mạng Neural
            let val = brain.predict(&board_flat.try_into().unwrap());

            if val > best_value {
                best_value = val;
                best_action = match action_dir {
                    Direction::Up => 0,
                    Direction::Down => 1,
                    Direction::Left => 2,
                    Direction::Right => 3,
                };
            }
        }
        best_action
    }

    pub fn train_step(&mut self, brain: &mut NTupleNetwork, action: u32, alpha: f64) -> (f64, f64) {
        // 1. Lấy dữ liệu TRƯỚC khi đi (tại trạng thái S)
        // ---------------------------------------------------
        let s_flat_vec = self.get_board_flat();
        let s_flat: [u32; 16] = s_flat_vec.try_into().unwrap_or([0u32; 16]);

        // A. Tính V(S): Giá trị dự đoán của Mạng Neural (Cái cần update)
        let v_s = brain.predict(&s_flat);

        // B. Tính Phi(S) & Score cũ: Dùng cho Reward Shaping
        // Lưu ý: get_composite_potential là hàm heuristic "lẩu thập cẩm" bác tự viết
        let phi_old = get_composite_potential(&self.game.board, &self.config);
        let score_old = self.game.score;

        // 2. Thực hiện hành động (Môi trường chuyển sang S')
        // ---------------------------------------------------
        let (_, _, done, _) = self.step(action);

        // 3. Lấy dữ liệu SAU khi đi (tại trạng thái S')
        // ---------------------------------------------------
        let phi_new = get_composite_potential(&self.game.board, &self.config);
        let score_new = self.game.score;

        // 4. Tính toán Adaptive Reward (Phần quan trọng nhất)
        // ---------------------------------------------------

        // Base Reward: Điểm thực tế kiếm được
        let base_reward = (score_new - score_old) as f64;

        // Raw Shaping: Chênh lệch tiềm năng (có nhân Gamma)
        // Gamma phải khớp với gamma của Brain (thường là 0.99)
        let gamma = self.gamma;
        let raw_shaping = (gamma * phi_new) - phi_old;

        // CÂN BẰNG ĐỘNG: Gọi struct AdaptiveReward để tính Final Reward
        // Giả sử bác đã khai báo self.adaptive trong struct ThreesEnv
        let final_reward = self
            .adaptive_manager
            .update_and_scale(base_reward, raw_shaping);

        // 5. Tính V(S') (TD Target)
        // ---------------------------------------------------
        let v_s_next = if done {
            0.0
        } else {
            let s_next_vec = self.get_board_flat();
            let s_next_flat: [u32; 16] = s_next_vec.try_into().unwrap_or([0u32; 16]);
            brain.predict(&s_next_flat)
        };

        // 6. Tính TD Error & Update Weights
        // ---------------------------------------------------

        // Công thức TD: Error = (Reward + Gamma * V(S')) - V(S)
        let td_error = final_reward + gamma * v_s_next - v_s;

        // Chia nhỏ error cho số lượng Tuple để tránh over-correction
        let num_tuples = brain.tuples.len() as f64;
        let split_delta = (td_error * alpha) / num_tuples; // Dùng biến alpha tham số
        brain.update_weights(&s_flat, split_delta);

        (td_error.abs(), final_reward)
    }

    pub fn get_symmetries(&self) -> Vec<[u32; 16]> {
        let boards = self.game.get_symmetries();
        let mut result = Vec::with_capacity(8);

        for board in boards.into_iter() {
            let mut flat = [0u32; 16];
            for r in 0..4 {
                for c in 0..4 {
                    flat[r * 4 + c] = board[r][c].value;
                }
            }
            result.push(flat);
        }
        result
    }

    pub fn get_board_flat(&self) -> [u32; 16] {
        // Ủy quyền trực tiếp cho Game xử lý
        self.game.get_board_flat()
    }

    fn score(&self) -> f64 {
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
    fn step(&mut self, actions: Vec<u32>) -> (Vec<Vec<u32>>, Vec<f64>, Vec<bool>) {
        // Parallel execution using Rayon
        // map collect is efficient
        let results: Vec<(Vec<u32>, f64, bool)> = self
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
                    let delta = new_score as f64 - old_score as f64;
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
