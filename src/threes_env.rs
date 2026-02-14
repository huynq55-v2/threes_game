use crate::game::{Direction, Game};
use crate::n_tuple_network::NTupleNetwork;
use crate::pbt::TrainingConfig;
use rand::seq::IndexedRandom;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[derive(Clone)]
pub struct ThreesEnv {
    pub game: Game,

    pub gamma: f64,
    pub config: TrainingConfig,

    pub traces: Vec<Vec<f64>>,
    pub active_trace_indices: Vec<Vec<usize>>,
}

impl ThreesEnv {
    pub fn new(gamma: f64) -> Self {
        let game = Game::new();
        ThreesEnv {
            game,
            gamma: gamma,
            config: TrainingConfig::default(),
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

    pub fn get_random_valid_action(&self) -> Option<Direction> {
        // 1. Thu thập các hướng đi hợp lệ vào một Vector
        let mut valid_actions = Vec::new();

        // Duyệt qua các enum variants (nếu bạn có danh sách variants thì dùng, không thì match 0..4)
        for &dir in &[
            Direction::Up,
            Direction::Down,
            Direction::Left,
            Direction::Right,
        ] {
            if self.game.can_move(dir) {
                valid_actions.push(dir);
            }
        }

        // 2. Kiểm tra nếu không có nước đi nào hợp lệ
        if valid_actions.is_empty() {
            return None;
        }

        // 3. Chọn ngẫu nhiên một action và trả về chính nó (không ép kiểu u32)
        let mut rng = rand::rng();

        // choose() trả về Option<&Direction>, ta dùng copied() để lấy Direction
        valid_actions.choose(&mut rng).copied()
    }

    // 1. ROOT NODE: Bắt đầu tìm kiếm
    pub fn get_best_action_depth(
        &self,
        brain: &NTupleNetwork,
        depth: u32,
    ) -> (Option<Direction>, f64) {
        let mut best_val = f64::NEG_INFINITY;
        let mut best_action: Option<Direction> = None;
        let directions = [
            Direction::Up,
            Direction::Down,
            Direction::Left,
            Direction::Right,
        ];

        for &dir in &directions {
            if !self.game.can_move(dir) {
                continue;
            }

            // Tạo Afterstate (Lớp 1)
            let after_game = self.game.gen_afterstate(dir);

            // Quyết định: Nếu hết depth thì dừng, còn không thì xuống Chance Node
            let val;
            if depth <= 1 {
                val = brain.predict_game(&after_game);
            } else {
                // Xuống lớp Chance, giảm depth
                val = self.search_chance_node(&after_game, depth - 1, brain);
            }

            if best_action.is_none() || val > best_val {
                best_val = val;
                best_action = Some(dir);
            }
        }

        (best_action, best_val)
    }

    fn search_chance_node(&self, after_state: &Game, depth: u32, brain: &NTupleNetwork) -> f64 {
        let outcomes = after_state.gen_all_possible_outcomes();

        if outcomes.is_empty() {
            return -1_000_000.0;
        }

        let mut total_expected_score = 0.0;
        let mut sum_prob = 0.0; // Biến dùng để chứng minh

        for (mut outcome_game, prob_outcome) in outcomes {
            // Cộng dồn xác suất để kiểm tra
            sum_prob += prob_outcome;

            // Đồng bộ Hints cho lớp dưới
            outcome_game.hints = outcome_game
                .predicted_future_distribution
                .iter()
                .map(|(val, _)| *val)
                .collect();

            let val = self.search_move_node(&outcome_game, depth, brain);
            total_expected_score += prob_outcome * val;
        }

        // Kiểm tra tính hội tụ (Floating point epsilon)
        if (sum_prob - 1.0).abs() > 1e-9 {
            panic!("Xác suất không bằng 1! Sum = {}", sum_prob);
        }

        total_expected_score
    }

    // 3. MOVE NODE: Chọn nước đi tốt nhất từ Full State
    fn search_move_node(&self, game: &Game, depth: u32, brain: &NTupleNetwork) -> f64 {
        let mut best_val = f64::NEG_INFINITY;
        let mut can_move = false;
        let directions = [
            Direction::Up,
            Direction::Down,
            Direction::Left,
            Direction::Right,
        ];

        for &dir in &directions {
            if game.can_move(dir) {
                can_move = true;

                let next_after_game = game.gen_afterstate(dir);
                let val;

                if depth <= 1 {
                    val = brain.predict_game(&next_after_game);
                } else {
                    val = self.search_chance_node(&next_after_game, depth - 1, brain);
                }

                if val > best_val {
                    best_val = val;
                }
            }
        }

        if !can_move {
            return -1_000_000.0;
        }
        best_val
    }

    // --- 1. ENTRY POINT: CHẠY SONG SONG (Dùng hàm này để Eval) ---
    pub fn get_best_action_depth_parallel(
        &self,
        brain: &NTupleNetwork,
        depth: u32,
    ) -> (Option<Direction>, f64) {
        let mut best_val = f64::NEG_INFINITY;
        let mut best_action: Option<Direction> = None;
        let mut can_move_any = false;

        let current_score = self.game.score;

        let directions = [
            Direction::Up,
            Direction::Down,
            Direction::Left,
            Direction::Right,
        ];

        // Duyệt tuần tự 4 hướng ở Root (vì chỉ có 4 cái, par_iter ko hiệu quả lắm)
        // Nhưng bên trong loop này sẽ gọi hàm xử lý song song
        for &dir in &directions {
            if self.game.can_move(dir) {
                can_move_any = true;

                // 1. Tạo Afterstate
                let next_after_game = self.game.gen_afterstate(dir);

                // 2. Tính Reward tức thì (Quan trọng!)
                let reward = next_after_game.score - current_score;

                // 3. Tính Future Value (SONG SONG HÓA Ở ĐÂY)
                let future_val = if depth <= 1 {
                    brain.predict_game(&next_after_game)
                } else {
                    // Gọi hàm Chance Node đặc biệt có Rayon
                    self.search_chance_node_parallel(&next_after_game, depth - 1, brain)
                };

                let total_val = reward + future_val;

                // Logic chọn best action
                if best_action.is_none() || total_val > best_val {
                    best_val = total_val;
                    best_action = Some(dir);
                }
            }
        }

        if !can_move_any {
            return (None, -1_000_000.0);
        }

        (best_action, best_val)
    }

    // --- 2. PARALLEL CHANCE NODE (Chỉ dùng cho lớp trên cùng) ---
    fn search_chance_node_parallel(
        &self,
        after_state: &Game,
        depth: u32,
        brain: &NTupleNetwork,
    ) -> f64 {
        let outcomes = after_state.gen_all_possible_outcomes();

        if outcomes.is_empty() {
            return -1_000_000.0;
        }

        // --- RAYON PARALLEL ITERATOR ---
        // Sử dụng par_iter để chia nhỏ các outcomes cho 8 core xử lý
        // map: tính toán từng nhánh
        // sum: cộng dồn kết quả lại (Expectation)
        let total_expected_score: f64 = outcomes
            .into_par_iter()
            .map(|(mut outcome_game, prob_outcome)| {
                // Logic bên trong chạy trên Thread riêng biệt

                // A. Đồng bộ Hints (từ distribution dự đoán)
                outcome_game.hints = outcome_game
                    .predicted_future_distribution
                    .iter()
                    .map(|(val, _)| *val)
                    .collect();

                // B. Gọi xuống Move Node (Tuần tự - Sequential)
                // Tại sao tuần tự? Vì ta đã chia nhỏ task ở đây rồi,
                // chia nhỏ tiếp sẽ gây overhead quản lý thread.
                let val = self.search_move_node(&outcome_game, depth, brain);

                // Trả về giá trị có trọng số
                prob_outcome * val
            })
            .sum(); // Rayon tự động cộng dồn song song cực nhanh

        total_expected_score
    }

    pub fn train_step(
        &mut self,
        brain: &mut NTupleNetwork,
        action: Direction,
        alpha: f64,
    ) -> (f64, f64) {
        if !self.game.can_move(action) {
            panic!("Invalid action");
        }

        let after_game = self.game.gen_afterstate(action);

        // Flatten board để đưa vào mạng
        let mut s_after_flat = [0u32; 16];
        for r in 0..4 {
            for c in 0..4 {
                s_after_flat[r * 4 + c] = after_game.board[r][c].value;
            }
        }

        // 2. Dự đoán V(S_after) từ mạng
        let v_after = brain.predict(&s_after_flat);

        // 3. Thực hiện hành động thật (Environment Step)
        let score_old = self.game.calculate_score();
        self.game.make_full_move(action);
        let done = self.game.is_game_over();
        let score_new = after_game.calculate_score();
        let base_reward = (score_new - score_old) as f64;

        // 4. Tính Target V(S'_after) cho bước tiếp theo
        // QUAN TRỌNG: Luôn dùng Ply = 1 (1-step lookahead) để tính Target chuẩn xác
        let v_next_after = if done {
            0.0
        } else {
            let (best_action, best_val) = self.get_best_action_depth(brain, 1);

            if best_action.is_none() {
                0.0
            } else {
                best_val
            }
        };

        // 5. Tính TD Error
        let td_error_raw = base_reward + self.gamma * v_next_after - v_after;

        // 5.5. GRADIENT CLIPPING: Kẹp TD Error để tránh nổ gradient
        // Khi gộp số to (192, 384...), Reward có thể vọt lên hàng trăm/nghìn,
        // gây ra TD Error cực lớn -> cập nhật weights quá mạnh -> mạng dao động.
        // Kẹp trong khoảng [-50, 50] giúp weights hội tụ mượt hơn.
        let td_error = td_error_raw.clamp(-50.0, 50.0);

        // 6. Cập nhật Traces & Weights
        let effective_alpha = alpha / brain.tuples.len() as f64;

        // Lazy init traces
        if self.traces.is_empty() {
            let sizes: Vec<usize> = brain.weights.iter().map(|w| w.len()).collect();
            self.reset_traces(brain.weights.len(), &sizes);
        }

        brain.update_weights_td_lambda(
            &mut self.traces,
            &mut self.active_trace_indices,
            &s_after_flat,
            td_error,
            effective_alpha,
        );

        (td_error_raw.abs(), base_reward)
    }

    pub fn get_board_flat(&self) -> [u32; 16] {
        self.game.get_board_flat()
    }
}
