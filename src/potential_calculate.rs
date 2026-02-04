use crate::tile::Tile;

pub fn calculate_disorder(board: &[[Tile; 4]; 4]) -> f32 {
    let mut total_penalty = 0.0;

    for r in 0..4 {
        for c in 0..4 {
            let val = board[r][c].value;
            if val == 0 {
                continue;
            }

            // Helper: 1,2 -> Rank 0; 3->1; 6->2 ...
            let get_calc_rank = |v: u32| -> i32 {
                if v <= 2 {
                    0
                } else {
                    ((v as f32 / 3.0).log2() as i32) + 1
                }
            };

            let rank_curr = get_calc_rank(val);

            // Helper check neighbor
            let check_neighbor = |n_val: u32| -> f32 {
                if n_val == 0 {
                    return 0.0;
                } // Cạnh ô trống thì ok, ko phạt

                // 1. Logic 1 và 2 (Giữ nguyên)
                if val <= 2 && n_val <= 2 {
                    if val == n_val {
                        return 27.0;
                    }
                    // 1-1 hoặc 2-2: Phạt nặng
                    else {
                        return 0.0;
                    } // 1-2: Ngon
                }

                // Logic số thường với 1,2 (Ví dụ 3 đứng cạnh 1) -> Rank 1 vs 0 -> Diff 1 -> OK
                if (val <= 2 && n_val > 2) || (val > 2 && n_val <= 2) {
                    let rank_n = get_calc_rank(n_val);
                    let diff = (rank_curr - rank_n).abs();
                    // Phạt nhẹ thôi vì số nhỏ dễ sửa
                    return 3.0_f32.powi(diff);
                }

                // 2. Logic High Rank (SỬA Ở ĐÂY)
                let rank_neighbor = get_calc_rank(n_val);
                let diff = (rank_curr - rank_neighbor).abs();

                // Công thức cũ: 3^diff
                // Công thức mới: (Rank_Current + Rank_Neighbor) * 3^diff
                // Ý nghĩa: Sai lầm ở Rank càng to, hình phạt càng nhân lên gấp bội.
                let magnitude_penalty = (rank_curr + rank_neighbor) as f32;

                magnitude_penalty * 3.0_f32.powi(diff)
            };

            // So sánh Phải & Dưới
            if c < 3 {
                total_penalty += check_neighbor(board[r][c + 1].value);
            }
            if r < 3 {
                total_penalty += check_neighbor(board[r + 1][c].value);
            }
        }
    }
    total_penalty
}

fn calculate_empty(board: &[[Tile; 4]; 4]) -> f32 {
    let mut count = 0;
    for r in 0..4 {
        for c in 0..4 {
            if board[r][c].value == 0 {
                count += 1;
            }
        }
    }
    count as f32
}

pub fn get_composite_potential(board: &[[Tile; 4]; 4]) -> f32 {
    let phi_empty = calculate_empty(board);
    let phi_disorder = calculate_disorder(board);

    let w_empty = 50.0; // Thưởng 50 điểm cho mỗi ô trống
    let w_disorder = 1.0; // Hệ số phạt disorder

    // Potential = Cái tốt - Cái xấu
    (w_empty * phi_empty) - (w_disorder * phi_disorder)
}
