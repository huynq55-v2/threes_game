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
        let mut valid_actions = Vec::new();

        // 1. Duyệt qua 4 hướng để thu thập các hướng đi được
        for action in 0..4 {
            let dir = match action {
                0 => Direction::Up,
                1 => Direction::Down,
                2 => Direction::Left,
                3 => Direction::Right,
                _ => continue,
            };

            // Dùng hàm can_move mà chúng ta đã thống nhất để kiểm tra
            if self.game.can_move(dir) {
                valid_actions.push(action);
            }
        }

        // 2. Nếu danh sách trống -> Hết đường đi
        if valid_actions.is_empty() {
            return 100; // Sentinel value báo hiệu Game Over
        }

        // 3. Chọn ngẫu nhiên một action trong danh sách hợp lệ
        let mut rng = rand::rng();
        // Dùng choose để bốc một phần tử ngẫu nhiên (An toàn hơn loop random)
        use rand::prelude::IndexedRandom;
        *valid_actions.choose(&mut rng).unwrap() as u32
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

    // pub fn get_best_action_expectimax_depth2(&self, brain: &NTupleNetwork) -> u32 {
    //     let mut best_val = f64::NEG_INFINITY;
    //     let mut best_action = 0;

    //     // --- PLY 1: MAX NODE (AI chọn hướng đi hiện tại) ---
    //     for action in 0..4 {
    //         let dir = match action {
    //             0 => Direction::Up,
    //             1 => Direction::Down,
    //             2 => Direction::Left,
    //             3 => Direction::Right,
    //             _ => continue,
    //         };

    //         // Lấy tất cả các kịch bản gạch mọc có thể xảy ra từ hướng đi này
    //         // Hàm này của Huy đã xử lý: Trượt -> Lấy Hint -> Nhân chéo Vị trí x Giá trị
    //         let outcomes = self.game.get_all_possible_outcomes(dir);

    //         if !outcomes.is_empty() {
    //             let mut expected_value_ply1 = 0.0;

    //             // Duyệt qua từng kịch bản gạch mọc (Xác suất chia đều như Huy nói)
    //             for outcome_game in &outcomes {
    //                 // --- PLY 2: MAX NODE (AI tìm phản xạ tốt nhất từ kịch bản đó - DEPTH 2) ---
    //                 // Thay vì chấm điểm tĩnh, ta dùng logic Afterstate để tìm nước đi "thoát hiểm" nhất
    //                 let best_future_score =
    //                     self.get_max_afterstate_score_from_game(outcome_game, brain);

    //                 expected_value_ply1 += best_future_score;
    //             }

    //             // Tính trung bình cộng xác suất (Huy: 100% cho 1,2,3 hoặc 33% cho Bonus)
    //             expected_value_ply1 /= outcomes.len() as f64;

    //             if expected_value_ply1 > best_val {
    //                 best_val = expected_value_ply1;
    //                 best_action = action;
    //             }
    //         }
    //     }
    //     best_action as u32
    // }

    // pub fn get_best_action_parallel_depth5(&self, brain: &NTupleNetwork) -> u32 {
    //     let actions = vec![0, 1, 2, 3];

    //     // Song song hóa 4 hướng đi chính
    //     let action_results: Vec<(u32, f64)> = actions
    //         .par_iter()
    //         .map(|&action| {
    //             let dir = match action {
    //                 0 => Direction::Up,
    //                 1 => Direction::Down,
    //                 2 => Direction::Left,
    //                 3 => Direction::Right,
    //                 _ => return (action, f64::NEG_INFINITY),
    //             };

    //             let outcomes = self.game.get_all_possible_outcomes(dir);
    //             if outcomes.is_empty() {
    //                 return (action, f64::NEG_INFINITY);
    //             }

    //             // Ở mỗi nhánh, ta gọi hàm đệ quy expectimax_node
    //             let total_val: f64 = outcomes
    //                 .iter()
    //                 .map(|outcome_game| {
    //                     self.expectimax_node(outcome_game, 4, brain) // depth 4 vì ply 1 đã xong
    //                 })
    //                 .sum();

    //             (action, total_val / outcomes.len() as f64)
    //         })
    //         .collect();

    //     // Lấy action có điểm kỳ vọng cao nhất
    //     action_results
    //         .into_iter()
    //         .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    //         .map(|(act, _)| act)
    //         .unwrap_or(0)
    // }

    // pub fn get_best_action_recursive(&self, brain: &NTupleNetwork, depth: u32) -> u32 {
    //     let mut best_val = -f64::MAX;
    //     let mut best_action = 0;

    //     // PLY 1: AI chọn nước đi đầu tiên
    //     for action in 0..4 {
    //         let dir = match action {
    //             0 => Direction::Up,
    //             1 => Direction::Down,
    //             2 => Direction::Left,
    //             3 => Direction::Right,
    //             _ => continue,
    //         };

    //         // Lấy danh sách các kịch bản gạch mọc (Chance Node 1)
    //         // (Hàm này của Huy đã tự update tracker và hint rồi)
    //         let outcomes = self.game.get_all_possible_outcomes(dir);

    //         if !outcomes.is_empty() {
    //             let mut expected_value = 0.0;

    //             // Tính giá trị trung bình của các kịch bản
    //             for outcome_game in &outcomes {
    //                 // Gọi đệ quy xuống các tầng sâu hơn
    //                 // depth = SEARCH_DEPTH - 1 vì ta đã đi xong nước đầu tiên
    //                 expected_value += self.expectimax_node(outcome_game, depth - 1, brain);
    //             }

    //             expected_value /= outcomes.len() as f64;

    //             if expected_value > best_val {
    //                 best_val = expected_value;
    //                 best_action = action;
    //             }
    //         }
    //     }

    //     // Fallback: Nếu không đường nào đi được (best_action vẫn = 0 và val âm vô cùng)
    //     // thì cứ trả về 0, game sẽ tự xử lý Game Over sau.
    //     best_action as u32
    // }

    // /// Hàm đệ quy xử lý các tầng tiếp theo
    // /// Input: game state đã mọc gạch xong (đang chờ AI đi nước tiếp theo)
    // fn expectimax_node(&self, game: &Game, depth: u32, brain: &NTupleNetwork) -> f64 {
    //     // BASE CASE: Nếu đã chạm đáy độ sâu
    //     // Ta dùng Afterstate (nhìn thêm 1 bước đệm) để đánh giá lá chắn cuối cùng
    //     if depth == 0 {
    //         return self.get_max_afterstate_score_from_game(game, brain);
    //     }

    //     let mut max_val = -f64::MAX;
    //     let mut can_move = false;

    //     // MAX NODE: AI chọn nước đi tốt nhất tại tầng này
    //     for action in 0..4 {
    //         let dir = match action {
    //             0 => Direction::Up,
    //             1 => Direction::Down,
    //             2 => Direction::Left,
    //             3 => Direction::Right,
    //             _ => continue,
    //         };

    //         // CHANCE NODE: Sinh ra các kịch bản gạch mọc tiếp theo
    //         let outcomes = game.get_all_possible_outcomes(dir);

    //         if !outcomes.is_empty() {
    //             can_move = true;
    //             let mut current_dir_expected_val = 0.0;

    //             for outcome_game in &outcomes {
    //                 // ĐỆ QUY TIẾP: Xuống tầng sâu hơn (depth - 1)
    //                 current_dir_expected_val +=
    //                     self.expectimax_node(outcome_game, depth - 1, brain);
    //             }

    //             // Tính trung bình (Average)
    //             current_dir_expected_val /= outcomes.len() as f64;

    //             // Maximize: AI sẽ chọn hướng có kỳ vọng cao nhất
    //             if current_dir_expected_val > max_val {
    //                 max_val = current_dir_expected_val;
    //             }
    //         }
    //     }

    //     // Nếu tại tầng này không đi được đâu nữa -> Game Over -> Điểm 0
    //     if !can_move {
    //         return 0.0;
    //     }

    //     max_val
    // }

    // fn get_max_afterstate_score_from_game(&self, game: &Game, brain: &NTupleNetwork) -> f64 {
    //     let mut max_val = f64::NEG_INFINITY;
    //     let mut can_move_any = false;

    //     // Thử 4 hướng phản xạ tiếp theo
    //     for action in 0..4 {
    //         let dir = match action {
    //             0 => Direction::Up,
    //             1 => Direction::Down,
    //             2 => Direction::Left,
    //             3 => Direction::Right,
    //             _ => continue,
    //         };

    //         // SỬ DỤNG TRỰC TIẾP HÀM CỦA HUY TẠI ĐÂY
    //         if let Some(after_board) = game.get_afterstate(dir) {
    //             can_move_any = true;

    //             // Chuyển board sang dạng phẳng để N-Tuple predict
    //             let mut flat = [0u32; 16];
    //             for r in 0..4 {
    //                 for c in 0..4 {
    //                     flat[r * 4 + c] = after_board[r][c].value;
    //                 }
    //             }

    //             let score = brain.predict(&flat);
    //             if score > max_val {
    //                 max_val = score;
    //             }
    //         }
    //     }

    //     // Nếu bước tiếp theo không đi được hướng nào (Game Over giả lập)
    //     if !can_move_any {
    //         return 0.0; // Phạt nặng vì nước đi trước đó dẫn đến ngõ cụt
    //     }

    //     max_val
    // }

    /// Hàm Wrapper: Gọi từ bên ngoài với số Ply mong muốn
    /// Ví dụ: ply = 1 (chỉ trượt), ply = 4 (trượt-mọc-trượt-mọc)
    pub fn get_best_action_ply(&self, brain: &NTupleNetwork, depth: u32) -> (u32, f64) {
        let mut best_val = f64::NEG_INFINITY;
        let mut best_action = 0;
        let mut found_valid_move = false;

        // Duyệt 4 hướng (Up, Down, Left, Right)
        for action in 0..4 {
            let dir = match action {
                0 => Direction::Up,
                1 => Direction::Down,
                2 => Direction::Left,
                3 => Direction::Right,
                _ => continue,
            };

            // PRUNING: Nếu không đi được, bỏ qua ngay
            if !self.game.can_move(dir) {
                continue;
            }

            // BƯỚC QUAN TRỌNG:
            let val = self.search_chance_node(&self.game, dir, depth, brain);

            // --- SỬA Ở ĐÂY ---
            // Logic cũ: if val > best_val { ... } -> SAI nếu val = -inf
            // Logic mới: Nếu đây là nước đi hợp lệ đầu tiên tìm thấy -> LẤY LUÔN
            // Hoặc nếu điểm cao hơn điểm cũ -> LẤY
            if !found_valid_move || val > best_val {
                best_val = val;
                best_action = action;
                found_valid_move = true; // Đánh dấu là đã có ít nhất 1 nước đi
            }
        }

        // Xử lý Game Over
        if !found_valid_move {
            // Trả về 0 và điểm phạt.
            // LƯU Ý: Bên ngoài gọi hàm này phải check game over trước hoặc xử lý điểm phạt này
            return (100, brain.predict_game(&self.game));
        }

        (best_action as u32, best_val)
    }

    fn search_chance_node(
        &self,
        game: &Game,
        dir: Direction,
        depth: u32,
        brain: &NTupleNetwork,
    ) -> f64 {
        // --- ĐIỂM DỪNG (LEAF NODE) ---
        // Nếu depth = 1, ta chỉ đánh giá Afterstate (Bàn cờ vừa trượt xong, chưa mọc quân)
        // Đây là kỹ thuật tối ưu tốc độ (nhìn ngắn 1 bước).
        if depth == 1 {
            // Vì root đã check can_move, hàm này CHẮC CHẮN trả về Some.
            // Dùng expect để khẳng định tính đúng đắn của logic.
            let after_board = game
                .get_afterstate(dir)
                .expect("🔥 BUG Logic: Root check can_move OK nhưng get_afterstate trả None");

            let mut flat = [0u32; 16];
            for i in 0..16 {
                flat[i] = after_board[i / 4][i % 4].value;
            }
            return brain.predict(&flat);
        }

        // --- SINH QUÂN (SPAWN) ---
        // Sinh ra các outcomes (Bàn cờ đã mọc quân mới)
        let outcomes = game.get_all_possible_outcomes_pure(dir);

        if outcomes.is_empty() {
            // Nếu không sinh được outcome nào dù can_move = true => Lỗi logic game nghiêm trọng
            unreachable!("🔥 BUG Logic: Move valid nhưng không sinh được outcome nào!");
        }

        // --- TÍNH TRUNG BÌNH (AVERAGE) ---
        let mut total_score = 0.0;
        let prob = 1.0 / outcomes.len() as f64; // Giả sử xác suất đều

        for outcome_game in &outcomes {
            if depth == 2 {
                // Nếu còn 2 ply (Move + Spawn), đây là tầng cuối cùng.
                // Đánh giá trực tiếp Game State sau khi mọc quân.
                total_score += prob * brain.predict_game(outcome_game);
            } else {
                // Nếu depth > 2, tiếp tục gọi đệ quy sang lượt người chơi (Max Node)
                // Giảm depth đi 2 (tương ứng với 1 cặp Move + Spawn đã hoàn thành)
                let val = self.search_max_node(outcome_game, depth - 2, brain);
                total_score += prob * val;
            }
        }

        total_score
    }

    fn search_max_node(&self, game: &Game, depth: u32, brain: &NTupleNetwork) -> f64 {
        // Điều kiện dừng đệ quy (thường không chạy vào đây nếu logic trên chuẩn, nhưng cứ để an toàn)
        if depth == 0 {
            return brain.predict_game(game);
        }

        let mut best_val = f64::NEG_INFINITY;
        let mut can_move_any = false;

        // Duyệt 4 hướng
        for action in 0..4 {
            let dir = match action {
                0 => Direction::Up,
                1 => Direction::Down,
                2 => Direction::Left,
                3 => Direction::Right,
                _ => continue,
            };

            // PRUNING: Cắt nhánh cụt
            if game.can_move(dir) {
                can_move_any = true;
                // Gọi quay lại Chance Node để xử lý tiếp
                let val = self.search_chance_node(game, dir, depth, brain);

                if val > best_val {
                    best_val = val;
                }
            }
        }

        // XỬ LÝ DEAD END (GAME OVER)
        if !can_move_any {
            // Nếu không đi được hướng nào, trả về điểm phạt.
            // Điểm phạt chính là điểm đánh giá của bàn cờ chết này (thường thấp).
            return brain.predict_game(game);
        }

        best_val
    }

    // pub fn get_best_action_afterstate(&self, brain: &NTupleNetwork) -> u32 {
    //     let mut best_action = 0;
    //     let mut best_val = f64::NEG_INFINITY;

    //     for action in 0..4 {
    //         let dir = match action {
    //             0 => Direction::Up,
    //             1 => Direction::Down,
    //             2 => Direction::Left,
    //             3 => Direction::Right,
    //             _ => continue,
    //         };

    //         if let Some(after_board) = self.game.get_afterstate(dir) {
    //             let mut flat = [0u32; 16];
    //             for r in 0..4 {
    //                 for c in 0..4 {
    //                     flat[r * 4 + c] = after_board[r][c].value;
    //                 }
    //             }

    //             let val = brain.predict(&flat);
    //             if val > best_val {
    //                 best_val = val;
    //                 best_action = action;
    //             }
    //         }
    //     }
    //     best_action as u32
    // }

    // /// Afterstate Recursive dùng để Gen Data
    // pub fn get_best_action_afterstate_recursive(&self, brain: &NTupleNetwork, depth: u32) -> u32 {
    //     let mut best_val = f64::NEG_INFINITY;
    //     let mut best_action = 0;

    //     for action in 0..4 {
    //         let dir = match action {
    //             0 => Direction::Up,
    //             1 => Direction::Down,
    //             2 => Direction::Left,
    //             3 => Direction::Right,
    //             _ => continue,
    //         };

    //         // Ply 1: AI thực hiện trượt board
    //         if let Some(after_board) = self.game.get_afterstate(dir) {
    //             let mut temp_game = self.game.clone();
    //             temp_game.board = after_board;

    //             // Đi sâu xuống các tầng tiếp theo (mỗi tầng là 1 cặp: Spawn + Move)
    //             let val = self.afterstate_node_recursive(&temp_game, depth - 1, brain);

    //             if val > best_val {
    //                 best_val = val;
    //                 best_action = action;
    //             }
    //         }
    //     }
    //     best_action as u32
    // }

    // fn afterstate_node_recursive(&self, game: &Game, depth: u32, brain: &NTupleNetwork) -> f64 {
    //     if depth <= 1 {
    //         return brain.predict_game(game);
    //     }

    //     let mut total_expected_val = 0.0;
    //     let mut move_count = 0;

    //     // Duyệt qua 4 hướng để tìm các kịch bản mọc gạch tiếp theo
    //     for action in 0..4 {
    //         let dir = match action {
    //             0 => Direction::Up,
    //             1 => Direction::Down,
    //             2 => Direction::Left,
    //             3 => Direction::Right,
    //             _ => continue,
    //         };

    //         // TRUYỀN DIR VÀO ĐÂY
    //         let outcomes = game.get_all_possible_outcomes_pure(dir);

    //         if outcomes.is_empty() {
    //             continue;
    //         }

    //         move_count += 1;
    //         let mut best_next_val = f64::NEG_INFINITY;

    //         for outcome_game in &outcomes {
    //             // Đệ quy xuống tầng sâu hơn
    //             let val = self.afterstate_node_recursive(outcome_game, depth - 1, brain);
    //             if val > best_next_val {
    //                 best_next_val = val;
    //             }
    //         }

    //         total_expected_val += if best_next_val == f64::NEG_INFINITY {
    //             0.0
    //         } else {
    //             best_next_val
    //         };
    //     }

    //     if move_count == 0 {
    //         return 0.0;
    //     }
    //     total_expected_val / move_count as f64
    // }

    pub fn train_step(&mut self, brain: &mut NTupleNetwork, action: u32, alpha: f64) -> (f64, f64) {
        // 1. Lấy Afterstate hiện tại (S_after)
        let dir = Direction::from_u32(action);

        let after_board = match self.game.get_afterstate(dir) {
            Some(board) => board,
            None => return (0.0, -10.0), // Trả về lỗi nếu action không hợp lệ
        };

        // Flatten board để đưa vào mạng
        let mut s_after_flat = [0u32; 16];
        for r in 0..4 {
            for c in 0..4 {
                s_after_flat[r * 4 + c] = after_board[r][c].value;
            }
        }

        // 2. Dự đoán V(S_after) từ mạng
        let v_after = brain.predict(&s_after_flat);

        // 3. Thực hiện hành động thật (Environment Step)
        let score_old = self.game.score;
        let (_, _, done, _) = self.step(action);
        let reward = (self.game.score - score_old) as f64;

        // 4. Tính Target V(S'_after) cho bước tiếp theo
        // QUAN TRỌNG: Luôn dùng Ply = 1 (1-step lookahead) để tính Target chuẩn xác
        let v_next_after = if done {
            0.0
        } else {
            // Gọi hàm tìm best action nhưng FIX CỨNG DEPTH = 1
            let (best_action, best_val) = self.get_best_action_ply(brain, 1);

            // Kiểm tra sentinel value 100 (tắc đường) hoặc giá trị vô cực
            if best_action == 100 || best_val == f64::NEG_INFINITY {
                0.0
            } else {
                best_val
            }
        };

        // 5. Tính TD Error
        let td_error = reward + self.gamma * v_next_after - v_after;

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
