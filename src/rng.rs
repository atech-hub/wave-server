//! Simple deterministic RNG (xorshift64).
//!
//! No external dependencies. Reproducible across platforms.

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Get current state for checkpointing.
    pub fn state(&self) -> u64 {
        self.state
    }

    /// Restore from checkpointed state.
    pub fn from_state(state: u64) -> Self {
        Self { state }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Uniform float in [0, 1)
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform float in [-limit, +limit]
    pub fn uniform(&mut self, limit: f32) -> f32 {
        (self.next_f32() * 2.0 - 1.0) * limit
    }

    /// Random index in [0, n)
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}
