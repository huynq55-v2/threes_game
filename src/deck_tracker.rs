use std::collections::HashMap;

// Các hằng số helper
fn get_value_from_rank(rank: u32) -> u32 {
    if rank == 0 {
        return 0;
    }
    3 * 2u32.pow(rank - 1)
}

#[derive(Clone)]
pub struct DeckTracker {
    // Túi 12 (Basic Bag) - Luôn chạy nền
    basic_counts: [u8; 4], // Index 1, 2, 3
    total_basic_remaining: u8,

    // Túi 21 (Bonus Bag) - Chỉ kích hoạt sau Move 21
    moves_since_start: u32,
    bonus_cycle_pos: u8,      // 1..21
    has_bonus_in_cycle: bool, // Chu kỳ này đã ra Bonus chưa?
}

impl DeckTracker {
    pub fn new() -> Self {
        Self {
            basic_counts: [0, 4, 4, 4],
            total_basic_remaining: 12,
            moves_since_start: 0,
            bonus_cycle_pos: 0, // 0 nghĩa là chưa kích hoạt
            has_bonus_in_cycle: false,
        }
    }

    pub fn update(&mut self, spawned_value: u32) {
        self.moves_since_start += 1;

        // 1. Cập nhật Túi Basic (1, 2, 3)
        // Dù là Bonus hay Basic thì khi ra 1,2,3 nó đều trừ vào kho Basic
        // (Theo cơ chế fallback: Bonus ra 1,2,3 vẫn tính là rút Basic)
        if spawned_value <= 3 {
            let idx = spawned_value as usize;
            if self.basic_counts[idx] > 0 {
                self.basic_counts[idx] -= 1;
                self.total_basic_remaining -= 1;
            }
            if self.total_basic_remaining == 0 {
                self.basic_counts = [0, 4, 4, 4];
                self.total_basic_remaining = 12;
            }
        }

        // 2. Cập nhật Túi Bonus (Chỉ sau move 21)
        if self.moves_since_start > 21 {
            // Nếu mới chuyển từ 21 sang 22, khởi tạo chu kỳ
            if self.bonus_cycle_pos == 0 {
                self.bonus_cycle_pos = 1;
                self.has_bonus_in_cycle = false;
            } else {
                self.bonus_cycle_pos += 1;
            }

            // Nếu ra quân >= 6, đánh dấu là đã nổ Bonus
            if spawned_value >= 6 {
                self.has_bonus_in_cycle = true;
            }

            // Hết chu kỳ 21, reset
            if self.bonus_cycle_pos > 21 {
                self.bonus_cycle_pos = 1;
                self.has_bonus_in_cycle = false;
            }
        }
    }

    /// Hàm dự đoán (Dùng cho Expectimax)
    pub fn predict_future(&self, max_tile_rank: u32) -> Vec<(u32, f64)> {
        let mut prob_map: HashMap<u32, f64> = HashMap::new();

        // A. Tính xác suất lượt này là lượt Bonus (P_bonus_slot)
        let mut p_bonus_slot = 0.0;

        // CHỈ KÍCH HOẠT KHI QUA 21 NƯỚC ĐẦU
        if self.moves_since_start > 21 {
            if self.has_bonus_in_cycle {
                p_bonus_slot = 0.0; // Đã ra rồi thì thôi
            } else {
                // Xác suất = 1 / số lượng slot còn lại trong chu kỳ
                let remaining = 21.0 - self.bonus_cycle_pos as f64 + 1.0;
                p_bonus_slot = 1.0 / remaining;
            }
        }

        // B. Logic Fallback (Code C# Unity)
        // int num = Mathf.Max(GetHighestRank() - 3, 0);
        let num = if max_tile_rank >= 3 {
            max_tile_rank - 3
        } else {
            0
        };

        // Điều kiện fallback: num < 2 (Tức là Max Tile < 48 / Rank 5)
        // Nếu fallback, Bonus slot biến thành lượt rút thường
        let is_fallback = num < 2;

        // C. Xác suất còn lại dành cho Basic
        let p_basic_total = if is_fallback {
            1.0 // Bonus slot bị hủy, dồn hết về Basic
        } else {
            1.0 - p_bonus_slot
        };

        // --- NHÁNH 1: BASIC TILES (1, 2, 3) ---
        if p_basic_total > 0.0 && self.total_basic_remaining > 0 {
            for val in 1..=3 {
                let count = self.basic_counts[val as usize];
                if count > 0 {
                    let p = (count as f64 / self.total_basic_remaining as f64) * p_basic_total;
                    *prob_map.entry(val).or_insert(0.0) += p;
                }
            }
        }

        // --- NHÁNH 2: BONUS TILES (>= 24) ---
        // Chỉ ra Bonus khi KHÔNG fallback và TRÚNG slot bonus
        if !is_fallback && p_bonus_slot > 0.0 {
            // Logic C#: if (num < 4) return GetValue(num);
            // else return Random(4, num + 1);

            // Lưu ý: num < 2 đã xử lý ở trên (fallback).
            // Ở đây chỉ còn num >= 2.

            if num < 4 {
                // Rank 5 (48) -> num=2 -> Value 6
                // Rank 6 (96) -> num=3 -> Value 12
                // NHƯNG logic C# của Huy bảo Random(4, ...).
                // Value 6 (rank 2) < 4? Đúng. Value 12 (rank 3) < 4? Đúng.
                // Vậy đoạn này vẫn trả về 6 hoặc 12 chuẩn.

                let val = get_value_from_rank(num);
                *prob_map.entry(val).or_insert(0.0) += p_bonus_slot;
            } else {
                // Max Rank >= 7 (192) -> num >= 4
                // Dải random từ Rank 4 (24) đến Rank num
                let start_rank = 4; // Rank 4 là con 24
                let end_rank = num;
                let count = (end_rank - start_rank + 1) as f64;

                let p_each = p_bonus_slot / count;
                for r in start_rank..=end_rank {
                    let val = get_value_from_rank(r);
                    *prob_map.entry(val).or_insert(0.0) += p_each;
                }
            }
        }

        prob_map.into_iter().collect()
    }
}
