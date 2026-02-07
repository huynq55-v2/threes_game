use crate::deck_tracker::DeckTracker;
use crate::game::{Direction, Game};
use crate::n_tuple_network::NTupleNetwork;
use crate::pbt::TrainingConfig;
use rand::Rng as _;

pub struct ThreesEnv {
    pub game: Game,
    gamma: f64,
    // adaptive_manager: AdaptiveManager, // Unused
    pub config: TrainingConfig,

    // THÊM 2 trường này vào Env (Local State)
    pub traces: Vec<Vec<f64>>,
    pub active_trace_indices: Vec<Vec<usize>>,
}

const SEARCH_DEPTH: u32 = 3;

impl ThreesEnv {
    pub fn new(gamma: f64) -> Self {
        let game = Game::new();
        ThreesEnv {
            game,
            gamma: gamma,
            // adaptive_manager: AdaptiveManager::new(),
            config: TrainingConfig::default(),

            // Khởi tạo rỗng
            traces: Vec::new(),
            active_trace_indices: Vec::new(),
        }
    }

    // Hàm mới để khởi tạo/reset traces
    pub fn reset_traces(&mut self, num_tables: usize, table_sizes: &[usize]) {
        // Nếu chưa khởi tạo hoặc kích thước sai -> Cấp phát lại
        if self.traces.len() != num_tables {
            self.traces.clear();
            self.active_trace_indices.clear();
            for &size in table_sizes {
                self.traces.push(vec![0.0; size]);
                self.active_trace_indices.push(Vec::with_capacity(1000));
            }
        } else {
            // Nếu đã có, chỉ cần clear giá trị active
            for (table_idx, indices) in self.active_trace_indices.iter_mut().enumerate() {
                for &feat_idx in indices.iter() {
                    self.traces[table_idx][feat_idx] = 0.0;
                }
                indices.clear();
            }
        }
    }

    pub fn set_config(&mut self, new_cfg: TrainingConfig) {
        self.config = new_cfg;
    }

    pub fn set_gamma(&mut self, gamma: f64) {
        self.gamma = gamma;
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

    pub fn get_best_action_expectimax_depth2(&self, brain: &NTupleNetwork) -> u32 {
        let mut best_val = f64::NEG_INFINITY;
        let mut best_action = 0;

        // --- PLY 1: MAX NODE (AI chọn hướng đi hiện tại) ---
        for action in 0..4 {
            let dir = match action {
                0 => Direction::Up,
                1 => Direction::Down,
                2 => Direction::Left,
                3 => Direction::Right,
                _ => continue,
            };

            // Lấy tất cả các kịch bản gạch mọc có thể xảy ra từ hướng đi này
            // Hàm này của Huy đã xử lý: Trượt -> Lấy Hint -> Nhân chéo Vị trí x Giá trị
            let outcomes = self.game.get_all_possible_outcomes(dir);

            if !outcomes.is_empty() {
                let mut expected_value_ply1 = 0.0;

                // Duyệt qua từng kịch bản gạch mọc (Xác suất chia đều như Huy nói)
                for outcome_game in &outcomes {
                    // --- PLY 2: MAX NODE (AI tìm phản xạ tốt nhất từ kịch bản đó - DEPTH 2) ---
                    // Thay vì chấm điểm tĩnh, ta dùng logic Afterstate để tìm nước đi "thoát hiểm" nhất
                    let best_future_score =
                        self.get_max_afterstate_score_from_game(outcome_game, brain);

                    expected_value_ply1 += best_future_score;
                }

                // Tính trung bình cộng xác suất (Huy: 100% cho 1,2,3 hoặc 33% cho Bonus)
                expected_value_ply1 /= outcomes.len() as f64;

                if expected_value_ply1 > best_val {
                    best_val = expected_value_ply1;
                    best_action = action;
                }
            }
        }
        best_action as u32
    }

    pub fn get_best_action_recursive(&self, brain: &NTupleNetwork) -> u32 {
        let mut best_val = -f64::MAX;
        let mut best_action = 0;

        // PLY 1: AI chọn nước đi đầu tiên
        for action in 0..4 {
            let dir = match action {
                0 => Direction::Up,
                1 => Direction::Down,
                2 => Direction::Left,
                3 => Direction::Right,
                _ => continue,
            };

            // Lấy danh sách các kịch bản gạch mọc (Chance Node 1)
            // (Hàm này của Huy đã tự update tracker và hint rồi)
            let outcomes = self.game.get_all_possible_outcomes(dir);

            if !outcomes.is_empty() {
                let mut expected_value = 0.0;

                // Tính giá trị trung bình của các kịch bản
                for outcome_game in &outcomes {
                    // Gọi đệ quy xuống các tầng sâu hơn
                    // depth = SEARCH_DEPTH - 1 vì ta đã đi xong nước đầu tiên
                    expected_value += self.expectimax_node(outcome_game, SEARCH_DEPTH - 1, brain);
                }

                expected_value /= outcomes.len() as f64;

                if expected_value > best_val {
                    best_val = expected_value;
                    best_action = action;
                }
            }
        }

        // Fallback: Nếu không đường nào đi được (best_action vẫn = 0 và val âm vô cùng)
        // thì cứ trả về 0, game sẽ tự xử lý Game Over sau.
        best_action as u32
    }

    /// Hàm đệ quy xử lý các tầng tiếp theo
    /// Input: game state đã mọc gạch xong (đang chờ AI đi nước tiếp theo)
    fn expectimax_node(&self, game: &Game, depth: u32, brain: &NTupleNetwork) -> f64 {
        // BASE CASE: Nếu đã chạm đáy độ sâu
        // Ta dùng Afterstate (nhìn thêm 1 bước đệm) để đánh giá lá chắn cuối cùng
        if depth == 0 {
            return self.get_max_afterstate_score_from_game(game, brain);
        }

        let mut max_val = -f64::MAX;
        let mut can_move = false;

        // MAX NODE: AI chọn nước đi tốt nhất tại tầng này
        for action in 0..4 {
            let dir = match action {
                0 => Direction::Up,
                1 => Direction::Down,
                2 => Direction::Left,
                3 => Direction::Right,
                _ => continue,
            };

            // CHANCE NODE: Sinh ra các kịch bản gạch mọc tiếp theo
            let outcomes = game.get_all_possible_outcomes(dir);

            if !outcomes.is_empty() {
                can_move = true;
                let mut current_dir_expected_val = 0.0;

                for outcome_game in &outcomes {
                    // ĐỆ QUY TIẾP: Xuống tầng sâu hơn (depth - 1)
                    current_dir_expected_val +=
                        self.expectimax_node(outcome_game, depth - 1, brain);
                }

                // Tính trung bình (Average)
                current_dir_expected_val /= outcomes.len() as f64;

                // Maximize: AI sẽ chọn hướng có kỳ vọng cao nhất
                if current_dir_expected_val > max_val {
                    max_val = current_dir_expected_val;
                }
            }
        }

        // Nếu tại tầng này không đi được đâu nữa -> Game Over -> Điểm 0
        if !can_move {
            return 0.0;
        }

        max_val
    }

    fn get_max_afterstate_score_from_game(&self, game: &Game, brain: &NTupleNetwork) -> f64 {
        let mut max_val = f64::NEG_INFINITY;
        let mut can_move_any = false;

        // Thử 4 hướng phản xạ tiếp theo
        for action in 0..4 {
            let dir = match action {
                0 => Direction::Up,
                1 => Direction::Down,
                2 => Direction::Left,
                3 => Direction::Right,
                _ => continue,
            };

            // SỬ DỤNG TRỰC TIẾP HÀM CỦA HUY TẠI ĐÂY
            if let Some(after_board) = game.get_afterstate(dir) {
                can_move_any = true;

                // Chuyển board sang dạng phẳng để N-Tuple predict
                let mut flat = [0u32; 16];
                for r in 0..4 {
                    for c in 0..4 {
                        flat[r * 4 + c] = after_board[r][c].value;
                    }
                }

                let score = brain.predict(&flat);
                if score > max_val {
                    max_val = score;
                }
            }
        }

        // Nếu bước tiếp theo không đi được hướng nào (Game Over giả lập)
        if !can_move_any {
            return 0.0; // Phạt nặng vì nước đi trước đó dẫn đến ngõ cụt
        }

        max_val
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
        // let _phi_old = get_composite_potential(&self.game.board, &self.config); // Giữ lại để tránh unused warning nếu cần, hoặc xóa

        // 2. Thực hiện hành động thật (Môi trường chuyển sang S')
        // Lúc này quái mới sinh ra, tạo thành S'
        let (_, _, done, _) = self.step(action);

        let score_new = self.game.score;
        // let _phi_new = get_composite_potential(&self.game.board, &self.config);

        let mut reward = (score_new - score_old) as f64;

        // if self.game.check_game_over() {
        //     reward -= 260.0;
        // }

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
        // brain.update_weights_td_lambda(&s_after_flat, td_error, effective_alpha);

        // KIỂM TRA & INIT TRACES NẾU CẦN (Lazy Init)
        if self.traces.is_empty() {
            let sizes: Vec<usize> = brain.weights.iter().map(|w| w.len()).collect();
            self.reset_traces(brain.weights.len(), &sizes);
        }

        // Gọi hàm update mới, truyền traces của Env vào Brain
        brain.update_weights_td_lambda(
            &mut self.traces,
            &mut self.active_trace_indices,
            &s_after_flat,
            td_error,
            effective_alpha,
        );

        (td_error.abs(), reward)
    }

    pub fn reset(&mut self) -> (Vec<u32>, Vec<u32>) {
        self.game = Game::new();

        // Reset traces khi game mới bắt đầu (nếu đã được init)
        if !self.traces.is_empty() {
            // Lưu ý: reset_traces ở đây chỉ clear giá trị, không re-alloc
            // Ta dùng logic clear nhanh đã viết trong reset_traces
            for (table_idx, indices) in self.active_trace_indices.iter_mut().enumerate() {
                for &feat_idx in indices.iter() {
                    self.traces[table_idx][feat_idx] = 0.0;
                }
                indices.clear();
            }
        }

        (self.get_board_flat().to_vec(), self.game.hints.clone())
    }

    pub fn get_board_flat(&self) -> [u32; 16] {
        // Ủy quyền trực tiếp cho Game xử lý
        self.game.get_board_flat()
    }
}
