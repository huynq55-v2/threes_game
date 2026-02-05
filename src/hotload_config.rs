use serde::Deserialize;

// Cấu hình đọc từ file (Dùng để can thiệp)
// Sử dụng Option: Nếu không có trong JSON hoặc để null, logic cũ sẽ được giữ nguyên
#[derive(Clone, Copy, Debug, Deserialize, Default)]
pub struct HotLoadConfig {
    pub w_empty_override: Option<f64>,
    pub w_disorder_override: Option<f64>,
    pub w_snake_override: Option<f64>,
    pub w_merge_override: Option<f64>,
    pub alpha_override: Option<f64>,
}
