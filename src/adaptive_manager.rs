pub struct AdaptiveManager {
    mean_base: f64,
    var_base: f64,
    mean_shaping: f64,
    var_shaping: f64,
    decay: f64,        // Thường là 0.99
    target_ratio: f64, // k = 0.2
}

impl AdaptiveManager {
    pub fn new() -> Self {
        Self {
            mean_base: 0.0,
            var_base: 1.0,
            mean_shaping: 0.0,
            var_shaping: 1.0,
            decay: 0.995,
            target_ratio: 0.2,
        }
    }

    pub fn update_and_scale(&mut self, base: f64, raw_shaping: f64) -> f64 {
        // Cập nhật stats cho Base
        let delta_b = base - self.mean_base;
        self.mean_base += (1.0 - self.decay) * delta_b;
        self.var_base =
            self.decay * self.var_base + (1.0 - self.decay) * delta_b * (base - self.mean_base);

        // Cập nhật stats cho Shaping
        let delta_s = raw_shaping - self.mean_shaping;
        self.mean_shaping += (1.0 - self.decay) * delta_s;
        self.var_shaping = self.decay * self.var_shaping
            + (1.0 - self.decay) * delta_s * (raw_shaping - self.mean_shaping);

        // Tính w
        let std_b = self.var_base.sqrt().max(0.0001);
        let std_s = self.var_shaping.sqrt().max(0.0001);

        let w = (self.target_ratio * (std_b / std_s)).min(10.0);

        base + (w * raw_shaping)
    }
}
