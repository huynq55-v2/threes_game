use std::cmp::max;
use std::collections::HashMap;

// Helper: Config game gốc
const K_SPECIAL_DEMOTION: u32 = 3;

pub fn get_value_from_rank(rank: u32) -> u32 {
    if rank == 0 {
        return 0;
    }
    3 * (1 << (rank - 1)) // Tương đương 3 * 2^(rank-1)
}

pub fn get_rank_from_value(value: u32) -> u32 {
    if value <= 2 {
        return 0;
    } // 1, 2 không có rank trong logic này
    if value == 3 {
        return 1;
    }
    // log2(value/3) + 1. Ví dụ 6 -> 2, 12 -> 3
    (value as f64 / 3.0).log2() as u32 + 1
}

#[derive(Clone)]
pub struct DeckTracker {
    // Túi 12 (Basic Bag)
    basic_counts: [u8; 4], // Index 1, 2, 3
    total_basic_remaining: u8,

    // Túi 21 (Bonus Bag)
    moves_since_start: u32,
    bonus_cycle_pos: u8,
    has_bonus_in_cycle: bool,
}

impl DeckTracker {
    pub fn new() -> Self {
        Self {
            basic_counts: [0, 4, 4, 4],
            total_basic_remaining: 12,
            moves_since_start: 0,
            bonus_cycle_pos: 0,
            has_bonus_in_cycle: false,
        }
    }

    pub fn update(&mut self, spawned_value: u32) {
        self.moves_since_start += 1;

        // 1. Logic trừ Basic Deck (Fallback mechanic)
        // Bất kể nguồn gốc (Basic hay Bonus), nếu ra 1,2,3 thì đều trừ vào túi 12
        if spawned_value <= 3 {
            let idx = spawned_value as usize;
            if self.basic_counts[idx] > 0 {
                self.basic_counts[idx] -= 1;
                self.total_basic_remaining -= 1;
            }
            // Reset túi 12 nếu hết
            if self.total_basic_remaining == 0 {
                self.basic_counts = [0, 4, 4, 4];
                self.total_basic_remaining = 12;
            }
        }

        // 2. Logic Bonus Cycle (21 moves)
        if self.moves_since_start > 21 {
            if self.bonus_cycle_pos == 0 {
                self.bonus_cycle_pos = 1;
                self.has_bonus_in_cycle = false;
            } else {
                self.bonus_cycle_pos += 1;
            }

            // Nếu ra tile >= 6 (Rank >= 2), chắc chắn là Bonus
            if spawned_value >= 6 {
                self.has_bonus_in_cycle = true;
            }

            if self.bonus_cycle_pos > 21 {
                self.bonus_cycle_pos = 1;
                self.has_bonus_in_cycle = false;
            }
        }
    }

    /// Hàm dự đoán tương lai (Đã sửa logic 2 tầng RNG)
    pub fn predict_future(&self, max_tile_rank: u32) -> Vec<(u32, f64)> {
        let mut prob_map: HashMap<u32, f64> = HashMap::new();

        // --- BƯỚC 1: TÍNH XÁC SUẤT CÁC SLOT (Basic vs Bonus) ---

        let mut p_bonus_slot = 0.0;

        // Chỉ kích hoạt Bonus Slot sau move 21
        if self.moves_since_start > 21 {
            if self.has_bonus_in_cycle {
                p_bonus_slot = 0.0; // Đã ra rồi thì các slot còn lại là Basic
            } else {
                let remaining_slots = 21.0 - self.bonus_cycle_pos as f64 + 1.0;
                p_bonus_slot = 1.0 / remaining_slots;
            }
        }

        // --- BƯỚC 2: XÁC ĐỊNH ANCHOR RANK (GetNextValue) ---

        // num = giới hạn trên của Rank có thể spawn
        // Logic Unity: Mathf.Max(GetHighestRank() - kSpecialDemotion, 0)
        let num = if max_tile_rank >= K_SPECIAL_DEMOTION {
            max_tile_rank - K_SPECIAL_DEMOTION
        } else {
            0
        };

        // Fallback Logic: Nếu Rank quá thấp (< 2), Bonus Slot bị hủy -> Biến thành Basic
        if num < 2 {
            // p_bonus_slot được chuyển sang cho Basic
            // p_basic_total = (1.0 - p_bonus) + p_bonus = 1.0
            p_bonus_slot = 0.0;
        }

        let p_basic_total = 1.0 - p_bonus_slot;

        // --- NHÁNH 1: BASIC DECK (1, 2, 3) ---
        if p_basic_total > 0.0 && self.total_basic_remaining > 0 {
            for val in 1..=3 {
                let count = self.basic_counts[val as usize];
                if count > 0 {
                    let p = (count as f64 / self.total_basic_remaining as f64) * p_basic_total;
                    *prob_map.entry(val).or_insert(0.0) += p;
                }
            }
        }

        // --- NHÁNH 2: BONUS DECK (2 Tầng RNG) ---
        if p_bonus_slot > 0.0 {
            // A. Xác định các Anchor Rank có thể xảy ra (futureValue)
            // Logic Unity:
            // if (num < 4) return GetValue(num); -> Chỉ có 1 anchor là 'num'
            // else return GetValue(Random.Range(4, num + 1)); -> Anchor từ 4 đến num

            let anchor_ranks: Vec<u32> = if num < 4 {
                vec![num] // num=2 hoặc num=3
            } else {
                (4..=num).collect() // num >= 4
            };

            // Xác suất chia đều cho các Anchor
            let p_per_anchor = p_bonus_slot / anchor_ranks.len() as f64;

            // B. Thực hiện Downgrade cho từng Anchor (DoSpawn)
            for &anchor_rank in &anchor_ranks {
                // Logic DoSpawn:
                // Range spawn: [max(2, rank - demotion + 1), rank]
                let min_spawn_rank = max(2, anchor_rank.saturating_sub(K_SPECIAL_DEMOTION) + 1);
                let max_spawn_rank = anchor_rank;

                let range_size = (max_spawn_rank - min_spawn_rank + 1) as f64;

                // Chia xác suất của Anchor này xuống các Tile con
                let p_final_per_tile = p_per_anchor / range_size;

                for r in min_spawn_rank..=max_spawn_rank {
                    let val = get_value_from_rank(r);
                    *prob_map.entry(val).or_insert(0.0) += p_final_per_tile;
                }
            }
        }

        prob_map.into_iter().collect()
    }
}
