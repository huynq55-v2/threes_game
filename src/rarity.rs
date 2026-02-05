#[derive(Debug, Clone)]
pub struct RarityEngine {
    // Đếm số lần xuất hiện của từng Rank (từ 0 đến 15)
    pub global_counts: [u64; 23],
    // Tổng số mẫu đã thu thập (bằng sum của mảng trên)
    pub total_seen: u64,
}

impl RarityEngine {
    pub fn new() -> Self {
        Self {
            global_counts: [0; 23],
            total_seen: 0,
        }
    }

    // [PHẦN BÁC THẤY THIẾU]: Hàm này gọi mỗi khi thấy 1 Tile xuất hiện (hoặc biến mất)
    pub fn register_observation(&mut self, rank: u8) {
        if rank != 0 {
            self.global_counts[rank as usize] += 1;
            self.total_seen += 1;
        }
    }

    // Hàm tính Reward bác đã có
    pub fn calculate_merge_reward(&self, rank: u8, local_board: &[u8; 16]) -> f64 {
        let local_count = local_board.iter().filter(|&&r| r == rank).count() as f64;

        // Công thức: Nghịch đảo xác suất (càng ít gặp càng to)
        // total_seen / counts[rank] chính là 1/P(rank)
        let global_factor =
            (self.total_seen as f64 + 1.0) / (self.global_counts[rank as usize] as f64 + 1.0);
        let local_factor = 16.0 / (local_count + 1.0);

        // Dynamic Part: Giá trị dựa trên độ hiếm (Trung bình Rank 1 ~ 3.3)
        let dynamic_part = global_factor.ln() * local_factor;

        // Base Part: ĐÃ SỬA TỪ 1.0 -> 3.0
        // Để cân bằng với dynamic_part của Rank 1
        dynamic_part + 3.0
    }
}
