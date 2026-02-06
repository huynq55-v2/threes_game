use crate::game::{Direction, Game};
use crate::n_tuple_network::NTupleNetwork;
use crate::pbt::TrainingConfig;
use crate::potential_calculate::get_composite_potential;
use rand::Rng as _;

pub struct ThreesEnv {
    pub game: Game,
    gamma: f64,
    // adaptive_manager: AdaptiveManager, // Unused
    pub config: TrainingConfig,
}

impl ThreesEnv {
    pub fn new(gamma: f64) -> Self {
        let game = Game::new();
        ThreesEnv {
            game,
            gamma: gamma,
            // adaptive_manager: AdaptiveManager::new(),
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

    pub fn get_best_action_afterstate(&self, brain: &NTupleNetwork) -> u32 {
        let mut best_action = 0;
        let mut best_val = f64::NEG_INFINITY;

        for action in 0..4 {
            let dir = match action {
                0 => Direction::Up,
                1 => Direction::Down,
                2 => Direction::Left,
                3 => Direction::Right,
                _ => continue,
            };

            if let Some(after_board) = self.game.get_afterstate(dir) {
                let mut flat = [0u32; 16];
                for r in 0..4 {
                    for c in 0..4 {
                        flat[r * 4 + c] = after_board[r][c].value;
                    }
                }

                let val = brain.predict(&flat);
                if val > best_val {
                    best_val = val;
                    best_action = action;
                }
            }
        }
        best_action as u32
    }

    pub fn train_step(&mut self, brain: &mut NTupleNetwork, action: u32, alpha: f64) -> (f64, f64) {
        // 1. Tính trạng thái Afterstate (S_after)
        let dir = match action {
            0 => Direction::Up,
            1 => Direction::Down,
            2 => Direction::Left,
            3 => Direction::Right,
            _ => unreachable!(),
        };

        // Lấy board sau khi di chuyển (chưa spawn)
        let after_board_opt = self.game.get_afterstate(dir);

        // Nếu nước đi không hợp lệ (về lý thuyết không nên xảy ra nếu đã check valid)
        if after_board_opt.is_none() {
            return (0.0, -10.0); // Phạt nặng
        }
        let after_board = after_board_opt.unwrap();

        // Flatten để đưa vào mạng
        let mut s_after_flat = [0u32; 16];
        for r in 0..4 {
            for c in 0..4 {
                s_after_flat[r * 4 + c] = after_board[r][c].value;
            }
        }

        // A. Tính V(S_after) - Đây là giá trị ta cần tối ưu
        let v_after = brain.predict(&s_after_flat);

        // Lưu điểm cũ để tính reward
        let score_old = self.game.score;
        let _phi_old = get_composite_potential(&self.game.board, &self.config); // Giữ lại để tránh unused warning nếu cần, hoặc xóa

        // 2. Thực hiện hành động thật (Môi trường chuyển sang S')
        // Lúc này quái mới sinh ra, tạo thành S'
        let (_, _, done, _) = self.step(action);

        let score_new = self.game.score;
        let _phi_new = get_composite_potential(&self.game.board, &self.config);

        let reward = (score_new - score_old) as f64;

        // 3. Tính V(S'_after) cho bước tiếp theo (TD Target)
        let v_next_after = if done {
            0.0
        } else {
            // Tìm nước đi tốt nhất tiếp theo từ S' (Greedy policy cho Target)
            // Lưu ý: Ở đây ta cần predict trên Afterstate của các nước đi tiếp theo
            let best_action_idx = self.get_best_action_afterstate(brain); // Cần viết thêm hàm này

            // Lấy giá trị Afterstate của nước đi tốt nhất đó
            let best_dir = match best_action_idx {
                0 => Direction::Up,
                1 => Direction::Down,
                2 => Direction::Left,
                3 => Direction::Right,
                _ => unreachable!(),
            };
            if let Some(next_after_board) = self.game.get_afterstate(best_dir) {
                let mut next_flat = [0u32; 16];
                for r in 0..4 {
                    for c in 0..4 {
                        next_flat[r * 4 + c] = next_after_board[r][c].value;
                    }
                }
                brain.predict(&next_flat)
            } else {
                0.0
            }
        };

        // 4. Tính TD Error & Update
        // Target hướng về Afterstate của bước sau
        let td_error = reward + self.gamma * v_next_after - v_after;

        // Update weights dựa trên Afterstate hiện tại (s_after_flat)
        let num_tuples = brain.tuples.len() as f64;
        let effective_alpha = alpha / num_tuples;

        // Update vào S_after, KHÔNG PHẢI S hiện tại
        brain.update_weights_td_lambda(&s_after_flat, td_error, effective_alpha);

        (td_error.abs(), reward)
    }

    pub fn get_board_flat(&self) -> [u32; 16] {
        // Ủy quyền trực tiếp cho Game xử lý
        self.game.get_board_flat()
    }
}
