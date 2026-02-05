use serde::Deserialize;

// 2. Cấu hình đọc từ file (Dùng để can thiệp)
#[derive(Clone, Copy, Debug, Deserialize)]
pub struct HotLoadConfig {
    pub w_empty_override: f64,
    pub w_disorder_override: f64,
    pub w_snake_override: f64,
    pub w_merge_override: f64,
    pub alpha_override: f64,
}

impl Default for HotLoadConfig {
    fn default() -> Self {
        Self {
            w_empty_override: -1.0,
            w_disorder_override: -1.0,
            w_snake_override: -1.0,
            w_merge_override: -1.0,
            alpha_override: -1.0,
        }
    }
}
