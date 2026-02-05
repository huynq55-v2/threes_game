use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const GOLDEN_RATIO: f64 = 1.61803398875;

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct TrainingConfig {
    #[serde(default)]
    pub w_empty: f64,
    #[serde(default)]
    pub w_snake: f64,

    // Thêm 2 em mới
    #[serde(default)]
    pub w_merge: f64, // Khởi tạo tầm 10.0
    #[serde(default)]
    pub w_disorder: f64, // Khởi tạo tầm 5.0
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            w_empty: 50.0,
            w_snake: 50.0,
            w_merge: 15.0,   // Khuyến khích gộp bài
            w_disorder: 5.0, // Phạt sự lộn xộn (số to cạnh số bé)
        }
    }
}

pub struct PBTManager {
    population: HashMap<u32, (f64, TrainingConfig)>,
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
        current_score: f64,
        current_config: TrainingConfig,
        buff_multiplier: f64,
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

        // 3. Logic Tiến Hóa: Nếu luồng hiện tại quá tệ so với luồng kém nhất (hoặc trung bình)
        // Ở đây bác dùng logic: Nếu điểm <= worst * 1.05 (tức là nằm trong nhóm kém) thì copy thằng giỏi nhất
        if current_score <= worst_score * 1.05 {
            let mut new_config = best_config;
            let mut rng = rand::rng();

            // --- MUTATION LOGIC ---

            // 1. Đột biến Empty
            if rng.random_bool(0.5) {
                new_config.w_empty *= buff_multiplier;
                new_config.w_empty = new_config.w_empty.clamp(0.1, f64::MAX);
            }

            // 2. Đột biến Snake
            if rng.random_bool(0.5) {
                new_config.w_snake *= buff_multiplier;
                new_config.w_snake = new_config.w_snake.clamp(0.1, f64::MAX);
            }

            // 3. Đột biến Merge (MỚI)
            if rng.random_bool(0.5) {
                new_config.w_merge *= buff_multiplier;
                new_config.w_merge = new_config.w_merge.clamp(0.1, f64::MAX);
            }

            // 4. Đột biến Disorder (MỚI)
            if rng.random_bool(0.5) {
                new_config.w_disorder *= buff_multiplier;
                new_config.w_disorder = new_config.w_disorder.clamp(0.1, f64::MAX);
            }

            println!(
                "🧬 [PBT] Thread {} TIẾN HÓA! Sc:{:.0} -> Emp:{:.1}, Snk:{:.1}, Mrg:{:.1}, Dis:{:.1}",
                thread_id, current_score, 
                new_config.w_empty, new_config.w_snake, new_config.w_merge, new_config.w_disorder
            );

            return (true, new_config);
        }

        (false, current_config)
    }

    // Thêm vào impl PBTManager
    pub fn get_best_config_entry(&self) -> Option<(f64, TrainingConfig)> {
        if self.population.is_empty() { return None; }
        
        let mut sorted_pop: Vec<_> = self.population.values().collect();
        sorted_pop.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        
        // Dùng HAI dấu sao (**) để lấy được giá trị thực tế
        Some(**sorted_pop.first().unwrap())
    }
}