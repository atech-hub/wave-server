//! Generation loop and sampling — the core inference engine.
//!
//! KV-cache enabled: prefill prompt once, then O(1) per new token.
//! Harmonic attention KV-cache is simpler than standard — phases are scalars.

use crate::model::{ModelWeights, KvCache};
use crate::rng::Rng;

use crate::prompt::Vocab;

/// Configuration for text generation.
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub repetition_penalty: Option<f32>,
}

/// Result of non-streaming generation.
pub struct GenerationResult {
    pub tokens: Vec<usize>,
    pub text: String,
}

/// Token event emitted during streaming generation.
pub struct TokenEvent {
    pub text: String,
    pub done: bool,
}

/// Memory offsets for injection into the forward pass.
/// Pre-scaled by alpha, ready to add to ODE initial conditions.
pub struct MemoryOffsets {
    pub offsets: Vec<(Vec<f32>, Vec<f32>)>, // [(r_offset, s_offset)] per ODE layer
}

/// Generate all tokens at once (non-streaming) with KV-cache.
pub fn generate(
    model: &ModelWeights,
    prompt_tokens: &[usize],
    config: &GenerationConfig,
    vocab: &Vocab,
    memory: Option<&MemoryOffsets>,
) -> GenerationResult {
    let _ = memory; // TODO: wire memory into cached forward
    let mut rng = make_rng();
    let mut tokens = prompt_tokens.to_vec();
    let mut generated = Vec::new();

    // Prefill: process entire prompt, populate KV-cache
    let mut cache = KvCache::new(&model.config);
    let mut logits = model.prefill(prompt_tokens, &mut cache);

    for _ in 0..config.max_tokens {
        if let Some(penalty) = config.repetition_penalty {
            if penalty != 1.0 {
                apply_repetition_penalty(&mut logits, &tokens, penalty);
            }
        }

        let token = sample_token(&logits, config, &mut rng);
        tokens.push(token);
        generated.push(token);

        // Decode next: O(1) per layer using KV-cache
        let pos = tokens.len() - 1;
        logits = model.forward_one_cached(token, pos, &mut cache);
    }

    let text = vocab.decode(&generated);
    GenerationResult { tokens: generated, text }
}

/// Generate with a custom forward pass function (for GPU dispatch).
pub fn generate_with_forward<F>(
    model: &ModelWeights,
    prompt_tokens: &[usize],
    config: &GenerationConfig,
    vocab: &Vocab,
    memory: Option<&MemoryOffsets>,
    mut forward_fn: F,
) -> GenerationResult
where
    F: FnMut(&[usize], Option<&[(&[f32], &[f32])]>) -> Vec<Vec<f32>>,
{
    let mut rng = make_rng();
    let block_size = model.config.block_size;
    let mut tokens = prompt_tokens.to_vec();
    let mut generated = Vec::new();

    let offset_slices: Option<Vec<(&[f32], &[f32])>> = memory.map(|m|
        m.offsets.iter().map(|(r, s)| (r.as_slice(), s.as_slice())).collect()
    );
    let mem_arg = offset_slices.as_deref();

    for _ in 0..config.max_tokens {
        let start = if tokens.len() > block_size { tokens.len() - block_size } else { 0 };
        let context = &tokens[start..];

        let logits_all = forward_fn(context, mem_arg);
        let mut logits = logits_all.last().unwrap().clone();

        if let Some(penalty) = config.repetition_penalty {
            if penalty != 1.0 {
                apply_repetition_penalty(&mut logits, &tokens, penalty);
            }
        }

        let token = sample_token(&logits, config, &mut rng);
        tokens.push(token);
        generated.push(token);
    }

    let text = vocab.decode(&generated);
    GenerationResult { tokens: generated, text }
}

/// Generate tokens one at a time with KV-cache, calling on_token for each.
/// Returns false from on_token to stop early (e.g. client disconnect).
pub fn generate_streaming<F>(
    model: &ModelWeights,
    prompt_tokens: &[usize],
    config: &GenerationConfig,
    vocab: &Vocab,
    memory: Option<&MemoryOffsets>,
    mut on_token: F,
) where
    F: FnMut(TokenEvent) -> bool,
{
    let _ = memory; // TODO: wire memory into cached forward
    let mut rng = make_rng();
    let mut tokens = prompt_tokens.to_vec();

    // Prefill: process entire prompt, populate KV-cache
    let mut cache = KvCache::new(&model.config);
    let mut logits = model.prefill(prompt_tokens, &mut cache);

    for i in 0..config.max_tokens {
        if let Some(penalty) = config.repetition_penalty {
            if penalty != 1.0 {
                apply_repetition_penalty(&mut logits, &tokens, penalty);
            }
        }

        let token = sample_token(&logits, config, &mut rng);
        tokens.push(token);

        let text = vocab.decode(&[token]);
        let done = i + 1 >= config.max_tokens;

        if !on_token(TokenEvent { text, done }) {
            break;
        }

        // Next token: O(1) per layer using KV-cache
        let pos = tokens.len() - 1;
        logits = model.forward_one_cached(token, pos, &mut cache);
    }
}

/// Sample a single token from logits using temperature + top-k + top-p.
fn sample_token(logits: &[f32], config: &GenerationConfig, rng: &mut Rng) -> usize {
    let temp = config.temperature.max(1e-8);

    // Temperature scaling
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temp).collect();

    // Collect indices (optionally top-k filtered)
    let mut candidates: Vec<usize> = (0..scaled.len()).collect();
    if let Some(k) = config.top_k {
        if k < candidates.len() {
            candidates.sort_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap());
            candidates.truncate(k);
        }
    }

    // Softmax over candidates
    let max_l = candidates.iter().map(|&i| scaled[i]).fold(f32::NEG_INFINITY, f32::max);
    let mut exp_vals: Vec<(usize, f32)> = candidates.iter()
        .map(|&i| (i, (scaled[i] - max_l).exp()))
        .collect();

    // Top-p (nucleus) filtering
    if config.top_p < 1.0 {
        exp_vals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let sum: f32 = exp_vals.iter().map(|(_, v)| v).sum();
        let threshold = config.top_p * sum;
        let mut cumsum = 0.0;
        let mut cutoff = exp_vals.len();
        for (i, (_, v)) in exp_vals.iter().enumerate() {
            cumsum += v;
            if cumsum >= threshold {
                cutoff = i + 1;
                break;
            }
        }
        exp_vals.truncate(cutoff);
    }

    // Categorical sample
    let sum: f32 = exp_vals.iter().map(|(_, v)| v).sum();
    let mut r = rng.next_f32() * sum;

    for &(idx, val) in &exp_vals {
        r -= val;
        if r <= 0.0 {
            return idx;
        }
    }

    exp_vals.last().unwrap().0
}

/// Penalize tokens that have already appeared in the sequence.
fn apply_repetition_penalty(logits: &mut [f32], tokens: &[usize], penalty: f32) {
    for &tok in tokens {
        if tok < logits.len() {
            if logits[tok] > 0.0 {
                logits[tok] /= penalty;
            } else {
                logits[tok] *= penalty;
            }
        }
    }
}

/// Create an RNG seeded from system time.
fn make_rng() -> Rng {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos() as u64;
    // Ensure non-zero seed (xorshift requires it)
    Rng::new(nanos | 1)
}
