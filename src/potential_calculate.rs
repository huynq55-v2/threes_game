use crate::{pbt::TrainingConfig, tile::Tile};

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

pub fn calculate_snake(board: &[[Tile; 4]; 4]) -> f32 {
    // Ma trận Snake chuẩn (Góc trên trái)
    // Các số này là lũy thừa để tạo độ dốc cực lớn:
    // M[0][0] là to nhất, sau đó đi dích dắc giảm dần
    // Bác có thể dùng công thức pow(4, k) hoặc đơn giản là fix cứng hằng số như này
    // để đỡ tốn CPU tính toán.

    // Pattern ZigZag:
    // 0  1  2  3  -> Giảm dần sang phải
    // 7  6  5  4  -> Giảm dần sang trái
    // 8  9 10 11  -> Giảm dần sang phải
    //15 14 13 12  -> Giảm dần sang trái

    // Trọng số càng to càng quan trọng (ở đây dùng đơn vị tuyến tính cho nhẹ,
    // nhưng khi nhân với Rank của Tile nó sẽ tạo ra hiệu ứng lũy thừa)
    const SNAKE_WEIGHTS: [f32; 16] = [
        1073741824.0,
        268435456.0,
        67108864.0,
        16777216.0, // Hàng 0: 4^15 -> 4^12
        65536.0,
        262144.0,
        1048576.0,
        4194304.0, // Hàng 1: 4^8  -> 4^11
        16384.0,
        4096.0,
        1024.0,
        256.0, // Hàng 2: 4^7  -> 4^4
        1.0,
        4.0,
        16.0,
        64.0, // Hàng 3: 4^0  -> 4^3
    ];

    // Chúng ta sẽ tính điểm cho 4 góc bằng cách xoay/đảo index truy cập
    // thay vì xoay bàn cờ (để tối ưu tốc độ)
    let mut max_score = 0.0;

    // 1. Góc Trên-Trái (Normal)
    let mut current_score = 0.0;
    for r in 0..4 {
        for c in 0..4 {
            let idx = r * 4 + c;
            let rank = get_rank(board[r][c].value);
            current_score += rank * SNAKE_WEIGHTS[idx];
        }
    }
    if current_score > max_score {
        max_score = current_score;
    }

    // 2. Góc Trên-Phải (Mirror Horizontal)
    current_score = 0.0;
    for r in 0..4 {
        for c in 0..4 {
            // Đảo cột: c -> 3-c
            let idx = r * 4 + c;
            let rank = get_rank(board[r][3 - c].value);
            current_score += rank * SNAKE_WEIGHTS[idx];
        }
    }
    if current_score > max_score {
        max_score = current_score;
    }

    // 3. Góc Dưới-Trái (Mirror Vertical)
    current_score = 0.0;
    for r in 0..4 {
        for c in 0..4 {
            // Đảo hàng: r -> 3-r
            let idx = r * 4 + c;
            let rank = get_rank(board[3 - r][c].value);
            current_score += rank * SNAKE_WEIGHTS[idx];
        }
    }
    if current_score > max_score {
        max_score = current_score;
    }

    // 4. Góc Dưới-Phải (Mirror Both)
    current_score = 0.0;
    for r in 0..4 {
        for c in 0..4 {
            let idx = r * 4 + c;
            let rank = get_rank(board[3 - r][3 - c].value);
            current_score += rank * SNAKE_WEIGHTS[idx];
        }
    }
    if current_score > max_score {
        max_score = current_score;
    }

    // Lưu ý: Có thể thêm 4 biến thể Transpose (xoay dọc) nữa nếu muốn "Rắn dọc"
    // Nhưng 4 góc ngang thường là đủ mạnh rồi.

    max_score
}

pub fn get_composite_potential(board: &[[Tile; 4]; 4], cfg: &TrainingConfig) -> f32 {
    let phi_empty = calculate_empty(board);
    // let phi_disorder = calculate_disorder(board);
    let phi_snake = calculate_snake(board); // <--- MỚI
                                            // Ép điểm Snake về tầm 0.0 -> 50.0 để PBT không hoảng hốt
    let normalized_snake = phi_snake / 1073741824.0;

    // Potential = (Empty * W) + (Snake * W) - (Disorder * W)
    // Lưu ý: Snake trả về số rất to (hàng triệu), nên W_Snake thường chỉ cần rất nhỏ (0.001 -> 1.0)

    (cfg.w_empty * phi_empty) + (cfg.w_snake * normalized_snake)
}

// Helper để lấy Rank (đã có ở bài trước)
fn get_rank(val: u32) -> f32 {
    if val <= 2 {
        0.0
    } else {
        (val as f32 / 3.0).log2() + 1.0
    }
}
