use rand::rng;
use rand::seq::SliceRandom;

/// A generalized "bag" or "deck" implementation for pseudo-random generation.
/// Ensures that within a cycle, items appear with a guaranteed frequency distribution.
#[derive(Debug, Clone)]
pub struct PseudoList<T> {
    template: Vec<T>,
    pool: Vec<T>,
    multiplier: u8,
}

impl<T> PseudoList<T>
where
    T: Clone,
{
    /// Creates a new PseudoList.
    /// `multiplier`: Determines how many copies of the template list are added to the pool.
    pub fn new(multiplier: u8) -> Self {
        Self {
            template: Vec::new(),
            pool: Vec::new(),
            multiplier,
        }
    }

    /// Adds an item to the template list (the definition of the deck).
    pub fn add(&mut self, item: T) {
        self.template.push(item);
    }

    /// Generates the pool based on the template and multiplier.
    /// This clears any existing pool!
    pub fn generate_list(&mut self) {
        self.pool.clear();
        for item in &self.template {
            for _ in 0..self.multiplier {
                self.pool.push(item.clone());
            }
        }
    }

    /// Shuffles the current pool.
    pub fn shuffle(&mut self) {
        let mut rng = rng();
        self.pool.shuffle(&mut rng);
    }

    /// Gets the next item from the pool.
    /// If the pool is empty, it automatically regenerates and shuffles (infinite deck).
    /// Returns None only if the template is empty.
    pub fn get_next(&mut self) -> Option<T> {
        if self.pool.is_empty() {
            // Refill needed
            if self.template.is_empty() {
                return None;
            }
            self.generate_list();
            self.shuffle();
        }
        self.pool.pop()
    }

    /// Helper to check if the pool needs regeneration.
    pub fn count(&self) -> usize {
        self.pool.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::threes_const::K_SPECIAL_RARENESS;

    #[test]
    fn test_pseudo_list_flow() {
        // Similar to the user's snippet
        let mut numbers = PseudoList::new(crate::threes_const::K_NUMBER_RANDOMNESS);
        numbers.add(1);
        numbers.add(2);
        numbers.add(3);

        // Template: [1, 2, 3]
        // Multiplier: 4
        // Expected pool size: 12 (4 ones, 4 twos, 4 threes)
        numbers.generate_list();
        assert_eq!(numbers.count(), 12);

        // Verify distribution
        let ones = numbers.pool.iter().filter(|&&x| x == 1).count();
        let twos = numbers.pool.iter().filter(|&&x| x == 2).count();
        let threes = numbers.pool.iter().filter(|&&x| x == 3).count();
        assert_eq!(ones, 4);
        assert_eq!(twos, 4);
        assert_eq!(threes, 4);

        numbers.shuffle();
        // It should still have 12 items after shuffle
        assert_eq!(numbers.count(), 12);

        // Consume one
        let val = numbers.get_next();
        assert!(val.is_some());
        assert_eq!(numbers.count(), 11);
    }

    #[test]
    fn test_special_list() {
        let mut special = PseudoList::new(1);
        special.add(1);
        for _ in 0..K_SPECIAL_RARENESS {
            special.add(0);
        }

        special.generate_list();
        special.shuffle();

        // 1 item of value 1, and 20 items of value 0. Total 21.
        let expected_count = 1 + K_SPECIAL_RARENESS as usize;
        assert_eq!(special.count(), expected_count);

        let ones_count = special.pool.iter().filter(|&&x| x == 1).count();
        let zeros_count = special.pool.iter().filter(|&&x| x == 0).count();

        assert_eq!(ones_count, 1);
        assert_eq!(zeros_count, K_SPECIAL_RARENESS as usize);
    }

    #[test]
    fn test_auto_refill() {
        let mut list = PseudoList::new(1);
        list.add(100);
        list.generate_list();

        // Initial state
        assert_eq!(list.count(), 1);

        // Pop last item
        let val = list.get_next();
        assert_eq!(val, Some(100));
        assert_eq!(list.count(), 0);

        // Next pop should trigger refill
        let val2 = list.get_next();
        assert_eq!(val2, Some(100));
        assert_eq!(list.count(), 0); // Refilled (size 1), then popped (size 0)
    }
}
