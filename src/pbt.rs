use rand::Rng;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct TrainingConfig {
    pub w_empty: f32,
    // pub w_disorder: f32,
    // Sau này thích thêm gì thì thêm vào đây, không sợ vỡ code
    // pub w_snake: f32,
    // pub w_merge: f32,
    pub w_snake: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            w_empty: 50.0,
            // w_disorder: 1.0,
            w_snake: 0.0,
        }
    }
}

pub struct PBTManager {
    population: HashMap<u32, (f32, TrainingConfig)>,
}

impl PBTManager {
    pub fn new() -> Self {
        Self {
            population: HashMap::new(),
        }
    }

    pub fn report_and_evolve(
        &mut self,
        thread_id: u32,
        current_score: f32,
        current_config: TrainingConfig,
    ) -> (bool, TrainingConfig) {
        // 1. Cập nhật kết quả
        self.population
            .insert(thread_id, (current_score, current_config));

        if self.population.len() < 4 {
            return (false, current_config);
        }

        // 2. Tìm Best & Worst
        let mut sorted_pop: Vec<_> = self.population.iter().collect();
        // Sort giảm dần (điểm cao lên đầu)
        sorted_pop.sort_by(|a, b| {
            b.1 .0
                .partial_cmp(&a.1 .0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best_config = sorted_pop.first().unwrap().1 .1;
        let worst_score = sorted_pop.last().unwrap().1 .0;

        // 3. Logic Tiến Hóa
        if current_score <= worst_score * 1.05 {
            let mut new_config = best_config;
            let mut rng = rand::rng();

            // --- MUTATION LOGIC ---

            // Đột biến w_empty
            if rng.random_bool(0.3) {
                new_config.w_empty *= rng.random_range(0.8..1.2);
                new_config.w_empty = new_config.w_empty.clamp(1.0, 500.0);
            }

            // Đột biến w_disorder
            // if rng.random_bool(0.3) {
            //     new_config.w_disorder *= rng.random_range(0.8..1.2);
            //     new_config.w_disorder = new_config.w_disorder.clamp(0.1, 20.0);
            // }

            // Đột biến w_snake (MỚI)
            // Snake rất mạnh nên cho phép range rộng hơn tí
            if rng.random_bool(0.3) {
                // Nếu đang bằng 0 thì kích hoạt nó lên số nhỏ
                if new_config.w_snake < 0.001 {
                    new_config.w_snake = rng.random_range(0.1..1.0);
                } else {
                    new_config.w_snake *= rng.random_range(0.8..1.2);
                }
                new_config.w_snake = new_config.w_snake.clamp(0.0, 1000.0);
            }

            println!(
                "🧬 [PBT] Thread {} TIẾN HÓA! Score:{:.0} -> Empty:{:.1}, Snake:{:.1}",
                thread_id, current_score, new_config.w_empty, new_config.w_snake
            );

            return (true, new_config);
        }

        (false, current_config)
    }
}
