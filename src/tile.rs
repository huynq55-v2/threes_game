use std::fmt;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct Tile {
    pub value: u32,
}

impl Tile {
    /// Creates a new Tile with a given value.
    pub fn new(value: u32) -> Self {
        Self { value }
    }

    /// Returns the rank of the tile based on the formula:
    /// Value 1, 2 -> Rank 0
    /// Value 3 -> Rank 1
    /// Value 6 -> Rank 2
    /// Value 12 -> Rank 3
    /// ...
    pub fn rank(&self) -> u8 {
        if self.value <= 2 {
            0
        } else {
            (self.value / 3).ilog2() as u8 + 1
        }
    }

    /// Checks if the tile is empty (value 0).
    pub fn is_empty(&self) -> bool {
        self.value == 0
    }
}

pub fn get_rank_from_value(value: u32) -> u8 {
    if value <= 2 {
        0
    } else {
        (value / 3).ilog2() as u8 + 1
    }
}

pub fn get_value_from_rank(rank: u8) -> u32 {
    // get value from rank if rank >= 1
    3 * 2u32.pow(rank as u32 - 1)
}

impl fmt::Display for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank_logic() {
        let t_empty = Tile::new(0);
        assert_eq!(t_empty.rank(), 0);

        let t1 = Tile::new(1);
        assert_eq!(t1.rank(), 0);

        let t2 = Tile::new(2);
        assert_eq!(t2.rank(), 0);

        let t3 = Tile::new(3);
        assert_eq!(t3.rank(), 1);

        let t6 = Tile::new(6);
        assert_eq!(t6.rank(), 2);

        let t12 = Tile::new(12);
        assert_eq!(t12.rank(), 3);

        let t24 = Tile::new(24);
        assert_eq!(t24.rank(), 4);

        let t48 = Tile::new(48);
        assert_eq!(t48.rank(), 5);
    }
}
