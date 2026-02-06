use crate::adaptive_manager::AdaptiveManager;
use crate::game::{Direction, Game};
use crate::n_tuple_network::NTupleNetwork;
use crate::pbt::TrainingConfig;
use crate::potential_calculate::get_composite_potential;
use pyo3::prelude::*;
use rand::Rng as _;
use rayon::prelude::*;

pub struct ThreesEnv {
    pub game: Game,
    gamma: f64,
    adaptive_manager: AdaptiveManager,
    pub config: TrainingConfig,
}

impl ThreesEnv {
    pub fn new(gamma: f64) -> Self {
        let game = Game::new();
        ThreesEnv {
            game,
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
        self.game = Game::new();

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

        // let phi_old = get_composite_potential(&self.game.board, &self.config);
        let score_old = self.game.score;

        if self.game.can_move(dir) {
            let (_, _) = self.game.move_dir(dir);

            let game_over = self.game.check_game_over();

            // let phi_new = get_composite_potential(&self.game.board, &self.config);
            let score_new = self.game.score;

            let base_reward = (score_new - score_old) as f64;
            // let gamma = self.gamma;
            // let raw_shaping = (gamma * phi_new) - phi_old;

            // let final_reward = self
            //     .adaptive_manager
            //     .update_and_scale(base_reward, raw_shaping);

            let final_reward = base_reward; // no shaping

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
        let mut max_of_min_quality = f64::NEG_INFINITY;

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
                    let future_value = brain.predict_game(outcome);
                    let move_quality = future_value;

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

        best_action as u32
    }

    pub fn get_best_action_expectimax(&self, brain: &NTupleNetwork) -> u32 {
        let mut best_val = -f64::MAX;
        let mut best_action = 0;

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
                    let future_value = brain.predict_game(outcome);
                    let move_quality = future_value;

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

    pub fn train_step(&mut self, brain: &mut NTupleNetwork, action: u32, alpha: f64) -> (f64, f64) {
        // 1. Lấy dữ liệu TRƯỚC khi đi (tại trạng thái S)
        // ---------------------------------------------------
        let s_flat_vec = self.get_board_flat();
        let s_flat: [u32; 16] = s_flat_vec.try_into().unwrap_or([0u32; 16]);

        // A. Tính V(S): Giá trị dự đoán của Mạng Neural (Cái cần update)
        let v_s = brain.predict(&s_flat);

        // B. Tính Phi(S) & Score cũ: Dùng cho Reward Shaping
        // Lưu ý: get_composite_potential là hàm heuristic "lẩu thập cẩm" bác tự viết
        // let phi_old = get_composite_potential(&self.game.board, &self.config);
        let score_old = self.game.score;

        // 2. Thực hiện hành động (Môi trường chuyển sang S')
        // ---------------------------------------------------
        let (_, _, done, _) = self.step(action);

        // 3. Lấy dữ liệu SAU khi đi (tại trạng thái S')
        // ---------------------------------------------------
        // let phi_new = get_composite_potential(&self.game.board, &self.config);
        let score_new = self.game.score;

        // 4. Tính toán Adaptive Reward (Phần quan trọng nhất)
        // ---------------------------------------------------

        let base_reward = (score_new - score_old) as f64;

        let gamma = self.gamma;
        // let raw_shaping = (gamma * phi_new) - phi_old;
        // let final_reward = self
        //     .adaptive_manager
        //     .update_and_scale(base_reward, raw_shaping);

        let final_reward = base_reward; // no shaping

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

    pub fn get_board_flat(&self) -> [u32; 16] {
        // Ủy quyền trực tiếp cho Game xử lý
        self.game.get_board_flat()
    }

    fn score(&self) -> f64 {
        self.game.score
    }
}
