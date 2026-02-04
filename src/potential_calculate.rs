use crate::{pbt::TrainingConfig, tile::Tile};

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

pub fn calculate_merge_potential(board: &[[Tile; 4]; 4]) -> f32 {
    let mut merges = 0.0;
    for r in 0..4 {
        for c in 0..4 {
            let val = board[r][c].value;
            if val == 0 {
                continue;
            }

            // Check bên phải
            if c < 3 {
                let right = board[r][c + 1].value;
                if can_merge(val, right) {
                    merges += 1.0;
                }
            }
            // Check bên dưới
            if r < 3 {
                let down = board[r + 1][c].value;
                if can_merge(val, down) {
                    merges += 1.0;
                }
            }
        }
    }
    merges // Càng nhiều càng tốt
}

pub fn calculate_disorder(board: &[[Tile; 4]; 4]) -> f32 {
    let mut penalty = 0.0;
    for r in 0..4 {
        for c in 0..4 {
            let val = board[r][c].value;
            if val == 0 {
                continue;
            }
            let rank_curr = get_rank(val);

            // Check neighbor (Right & Down)
            let check = |n_val: u32| -> f32 {
                if n_val == 0 {
                    return 0.0;
                }
                let rank_n = get_rank(n_val);

                // Tính độ chênh lệch Rank
                let diff = (rank_curr - rank_n).abs();

                // Nếu chênh lệch > 1 thì bắt đầu phạt lũy thừa
                if diff > 1.0 {
                    // Ví dụ: Rank 8 (384) cạnh Rank 1 (3) -> Diff 7 -> Phạt nặng
                    return diff.powf(2.5);
                }
                0.0
            };

            if c < 3 {
                penalty += check(board[r][c + 1].value);
            }
            if r < 3 {
                penalty += check(board[r + 1][c].value);
            }
        }
    }
    penalty // Càng thấp càng tốt (nên w_disorder sẽ là số âm hoặc trừ đi)
}

pub fn get_composite_potential(board: &[[Tile; 4]; 4], cfg: &TrainingConfig) -> f32 {
    let phi_empty = calculate_empty(board);
    let phi_snake = calculate_snake(board) / 1073741824.0; // Normalized Snake

    // 1. Thêm Merge Potential (Normalize nhẹ vì max khoảng 24 cặp)
    let phi_merge = calculate_merge_potential(board) / 10.0;

    // 2. Thêm Disorder (Normalize vì có thể rất to)
    // Chia 100 để nó về tầm 0.x -> 5.0
    let phi_disorder = calculate_disorder(board) / 100.0;

    // Tổng hợp:
    // Empty và Snake là CỘNG
    // Merge là CỘNG
    // Disorder là TRỪ (vì là hình phạt)
    (cfg.w_empty * phi_empty) + (cfg.w_snake * phi_snake) + (cfg.w_merge * phi_merge)
        - (cfg.w_disorder * phi_disorder)
}

// Helper để lấy Rank (đã có ở bài trước)
fn get_rank(val: u32) -> f32 {
    if val <= 2 {
        0.0
    } else {
        (val as f32 / 3.0).log2() + 1.0
    }
}

// Helper kiểm tra luật Threes!
fn can_merge(a: u32, b: u32) -> bool {
    if a == 0 || b == 0 {
        return false;
    } // Không tính ô trống là merge
    if (a == 1 && b == 2) || (a == 2 && b == 1) {
        return true;
    }
    if a > 2 && a == b {
        return true;
    }
    false
}
