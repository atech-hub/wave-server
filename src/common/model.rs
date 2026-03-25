//! Wave-engine model — CPU reference forward pass for inference.
//!
//! Architecture: parallel blocks (GPT-J formulation)
//!   x = x + attn(LN(x)) + FFN(LN(x))
//!
//! Attention: harmonic coherence scoring — cos(n * Δθ)
//! FFN: dual-maestro Kerr-ODE — maestro_in → ODE → maestro_out → out_proj
//! Embeddings: frozen harmonic table — cos(n*θ) / sin(n*θ)

use std::f32::consts::PI;

/// Global AGC for inference — matches wave-engine training regulation.
static AGC: std::sync::LazyLock<std::sync::Mutex<crate::common::agc::OdeAgc>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(crate::common::agc::OdeAgc::new()));

/// Apply AGC knee compression to preconditioned input before ODE.
fn agc_process(precond: &mut [f32], n_bands: usize) {
    // Collect magnitudes, observe, compress — same as engine
    let mags: Vec<f32> = (0..n_bands).map(|k| {
        let r = precond[k * 2];
        let s = precond[k * 2 + 1];
        (r * r + s * s).sqrt()
    }).collect();
    let mut agc = AGC.lock().unwrap();
    agc.observe(&mags);
    let threshold = agc.threshold;
    drop(agc); // release lock before compression loop
    for k in 0..n_bands {
        let r = precond[k * 2];
        let s = precond[k * 2 + 1];
        let mag = (r * r + s * s).sqrt();
        if mag > threshold && mag > 0.001 {
            let excess = mag - threshold;
            let compressed = threshold + threshold * (excess / threshold).tanh();
            let scale = compressed / mag;
            precond[k * 2] *= scale;
            precond[k * 2 + 1] *= scale;
        }
    }
}

// ─── Model configuration ──────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct ModelConfig {
    pub n_bands: usize,
    pub n_head: usize,
    pub n_layers: usize,
    pub maestro_dim: usize,
    pub block_size: usize,
    pub rk4_n_steps: usize,
}

impl ModelConfig {
    pub fn default_768() -> Self {
        Self {
            n_bands: 384,
            n_head: 12,
            n_layers: 24,
            maestro_dim: 16,
            block_size: 256,
            rk4_n_steps: 16,
        }
    }

    pub fn n_embd(&self) -> usize { self.n_bands * 2 }
    pub fn head_dim(&self) -> usize { self.n_embd() / self.n_head }

    pub fn validate(&self) {
        assert!(self.n_bands > 0);
        assert!(self.n_head > 0);
        assert_eq!(self.n_embd() % self.n_head, 0, "n_embd must be divisible by n_head");
        assert!(self.rk4_n_steps > 0);
    }
}

// ─── Weight structures ──────────────────────────────────────

#[derive(Clone)]
pub struct LinearWeights {
    pub w: Vec<Vec<f32>>,
    pub b: Vec<f32>,
}

#[derive(Clone)]
pub struct LayerNormWeights {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

#[derive(Clone)]
pub struct KerrWeights {
    pub gamma_raw: Vec<f32>,
    pub omega: Vec<f32>,
    pub alpha: f32,
    pub beta: f32,
    pub rk4_n_steps: usize,
}

#[derive(Clone)]
pub struct MaestroWeights {
    pub squeeze: LinearWeights,
    pub process_1: LinearWeights,
}

/// Block-diagonal linear — groups of bands processed independently.
#[derive(Clone)]
pub struct BlockDiagonalWeights {
    pub groups: Vec<LinearWeights>,  // n_groups × (group_size, group_size)
    pub n_groups: usize,
    pub group_size: usize,
}

#[derive(Clone)]
pub struct KerrDualMaestroWeights {
    pub kerr: KerrWeights,
    pub maestro_in: MaestroWeights,
    pub maestro_out: MaestroWeights,
    pub out_proj: BlockDiagonalWeights,
}

/// Weights for one harmonic coherence attention head.
#[derive(Clone)]
pub struct WaveAttnHeadWeights {
    pub harmonic_raw: f32,
    pub phase_proj_w: Vec<Vec<f32>>,  // [2, n_embd]
    pub phase_proj_b: Vec<f32>,       // [2]
    pub v_proj_w: Vec<Vec<f32>>,      // [head_dim, head_dim]
    pub v_proj_b: Vec<f32>,           // [head_dim]
}

/// Weights for full multi-head wave attention.
#[derive(Clone)]
pub struct WaveAttnWeights {
    pub heads: Vec<WaveAttnHeadWeights>,
    pub out_proj_w: Vec<Vec<f32>>,    // [n_embd, n_embd]
    pub out_proj_b: Vec<f32>,         // [n_embd]
}

/// Weights for one parallel wave block.
#[derive(Clone)]
pub struct WaveBlockWeights {
    pub ln: LayerNormWeights,
    pub ln_ffn: LayerNormWeights,
    pub attn: WaveAttnWeights,
    pub ffn: KerrDualMaestroWeights,
}

// ─── KV-Cache ──────────────────────────────────────────────

/// Per-layer KV cache for harmonic coherence attention.
/// "K" = phase angles (scalar per head), "V" = projected values (head_dim per head).
/// Simpler than standard transformer KV-cache because phases are scalars.
struct LayerKvCache {
    /// Cached phase angles: [n_head][n_cached_positions]
    phases: Vec<Vec<f32>>,
    /// Cached value projections: [n_head][n_cached_positions][head_dim]
    values: Vec<Vec<Vec<f32>>>,
    /// Cached bucket assignments: [n_head][n_cached_positions]
    buckets: Vec<Vec<usize>>,
}

/// Full KV-cache across all layers.
pub struct KvCache {
    layers: Vec<LayerKvCache>,
    /// Hidden states entering each layer (needed for FFN on new token).
    /// [n_layers][n_cached_positions][n_embd] — the hidden state before each block.
    hidden_per_layer: Vec<Vec<Vec<f32>>>,
    /// Number of positions currently cached.
    n_cached: usize,
}

impl KvCache {
    /// Create empty cache for a model.
    pub fn new(config: &ModelConfig) -> Self {
        let n_head = config.n_head;
        let n_layers = config.n_layers;
        Self {
            layers: (0..n_layers).map(|_| LayerKvCache {
                phases: vec![Vec::new(); n_head],
                values: vec![Vec::new(); n_head],
                buckets: vec![Vec::new(); n_head],
            }).collect(),
            hidden_per_layer: vec![Vec::new(); n_layers],
            n_cached: 0,
        }
    }

    /// Clear the cache (new conversation).
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            for h in &mut layer.phases { h.clear(); }
            for h in &mut layer.values { h.clear(); }
            for h in &mut layer.buckets { h.clear(); }
        }
        for h in &mut self.hidden_per_layer { h.clear(); }
        self.n_cached = 0;
    }

    pub fn len(&self) -> usize { self.n_cached }
}

/// Full model weights.
pub struct ModelWeights {
    pub config: ModelConfig,
    pub vocab_size: usize,
    pub wte: Vec<Vec<f32>>,   // [vocab_size, n_embd] — frozen harmonic table
    pub wpe: Vec<Vec<f32>>,   // [block_size, n_embd] — positional encoding
    pub blocks: Vec<WaveBlockWeights>,
    pub ln_f: LayerNormWeights,
    pub lm_head: Vec<Vec<f32>>, // [vocab_size, n_embd]
}

// ─── Primitive operations ──────────────────────────────────────

fn linear(w: &[Vec<f32>], b: &[f32], x: &[f32]) -> Vec<f32> {
    w.iter().zip(b.iter())
        .map(|(row, &bias)| bias + row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum::<f32>())
        .collect()
}

pub fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let std = (var + 1e-5).sqrt();
    (0..n).map(|i| (x[i] - mean) / std * weight[i] + bias[i]).collect()
}

fn block_diagonal_forward(weights: &BlockDiagonalWeights, x: &[f32]) -> Vec<f32> {
    let n_embd = weights.n_groups * weights.group_size;
    let mut out = vec![0.0f32; n_embd];
    for g in 0..weights.n_groups {
        let start = g * weights.group_size;
        let gw = &weights.groups[g];
        for i in 0..weights.group_size {
            let mut sum = gw.b[i];
            for j in 0..weights.group_size {
                sum += gw.w[i][j] * x[start + j];
            }
            out[start + i] = sum;
        }
    }
    out
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

// ─── Embedding ──────────────────────────────────────────────

/// Build frozen harmonic embedding table.
/// Find two coprime moduli near sqrt(vocab_size) whose product >= vocab_size.
/// Sexagenary principle: small incommensurate grids cover more than one large grid.
fn find_coprime_moduli(vocab_size: usize) -> (usize, usize) {
    fn gcd(mut a: usize, mut b: usize) -> usize {
        while b != 0 { let t = b; b = a % b; a = t; } a
    }
    let root = (vocab_size as f64).sqrt().ceil() as usize;
    let mut m1 = root;
    if m1 % 2 == 0 { m1 += 1; }
    let mut m2 = m1 + 2;
    while gcd(m1, m2) != 1 || m1 * m2 < vocab_size { m2 += 1; }
    (m1, m2)
}

/// Multi-grid harmonic embeddings — must match wave-engine/src/common/embed.rs exactly.
/// Grid 1 (half bands): tok mod m1 on m1-circle.
/// Grid 2 (half bands): tok mod m2 on m2-circle.
pub fn build_harmonic_table(vocab_size: usize, n_bands: usize) -> Vec<Vec<f32>> {
    let n_embd = n_bands * 2;
    let half = n_bands / 2;
    let (m1, m2) = find_coprime_moduli(vocab_size);
    eprintln!("  [embed] Multi-grid: m1={}, m2={}, lcm_coverage={}, vocab={}", m1, m2, m1 * m2, vocab_size);
    (0..vocab_size).map(|tok| {
        let mut emb = vec![0.0f32; n_embd];
        let theta1 = (tok % m1) as f32 * 2.0 * PI / m1 as f32;
        for n in 0..half {
            let phase = (n + 1) as f32 * theta1;
            emb[n * 2] = phase.cos();
            emb[n * 2 + 1] = phase.sin();
        }
        let theta2 = (tok % m2) as f32 * 2.0 * PI / m2 as f32;
        for n in 0..half {
            let idx = half + n;
            let phase = (n + 1) as f32 * theta2;
            emb[idx * 2] = phase.cos();
            emb[idx * 2 + 1] = phase.sin();
        }
        emb
    }).collect()
}

/// Build positional encoding table.
pub fn build_positional_table(block_size: usize, n_bands: usize) -> Vec<Vec<f32>> {
    let n_embd = n_bands * 2;
    (0..block_size).map(|pos| {
        let mut pe = vec![0.0f32; n_embd];
        for n in 0..n_bands {
            let freq = 1.0 / (10000.0f32).powf(2.0 * n as f32 / n_embd as f32);
            pe[n * 2] = (pos as f32 * freq).sin();
            pe[n * 2 + 1] = (pos as f32 * freq).cos();
        }
        pe
    }).collect()
}

// ─── Harmonic coherence attention ──────────────────────────────

fn project_phase(x: &[f32], proj_w: &[Vec<f32>], proj_b: &[f32]) -> f32 {
    let n_embd = x.len();
    let mut r = proj_b[0];
    let mut s = proj_b[1];
    for j in 0..n_embd {
        r += proj_w[0][j] * x[j];
        s += proj_w[1][j] * x[j];
    }
    s.atan2(r)
}

fn wave_attention_forward(
    weights: &WaveAttnWeights,
    x: &[Vec<f32>],
    n_bands: usize,
) -> Vec<Vec<f32>> {
    let t = x.len();
    let n_embd = n_bands * 2;
    let n_head = weights.heads.len();
    let head_dim = n_embd / n_head;

    let mut out = vec![vec![0.0f32; n_embd]; t];

    for head in 0..n_head {
        let harmonic_n = softplus(weights.heads[head].harmonic_raw);
        let offset = head * head_dim;

        // Precompute phase angles
        let phases: Vec<f32> = (0..t).map(|pos| {
            project_phase(&x[pos], &weights.heads[head].phase_proj_w, &weights.heads[head].phase_proj_b)
        }).collect();

        // Value projection
        let v_all: Vec<Vec<f32>> = (0..t).map(|pos| {
            let mut v = vec![0.0f32; head_dim];
            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for j in 0..head_dim {
                    sum += weights.heads[head].v_proj_w[d][j] * x[pos][offset + j];
                }
                v[d] = sum + weights.heads[head].v_proj_b[d];
            }
            v
        }).collect();

        // Phase-hashed sparse attention
        const N_BUCKETS: usize = 8;
        let bucket_width = std::f32::consts::TAU / N_BUCKETS as f32;

        let buckets: Vec<usize> = phases.iter().map(|&p| {
            let normalized = ((p % std::f32::consts::TAU) + std::f32::consts::TAU) % std::f32::consts::TAU;
            ((normalized / bucket_width) as usize).min(N_BUCKETS - 1)
        }).collect();

        let mut bucket_positions: Vec<Vec<usize>> = vec![Vec::new(); N_BUCKETS];
        for (pos, &b) in buckets.iter().enumerate() {
            bucket_positions[b].push(pos);
        }

        for qi in 0..t {
            let qi_bucket = buckets[qi];
            let mut scores = vec![f32::NEG_INFINITY; t];

            for db in 0..=2 {
                let target_bucket = if db == 0 {
                    (qi_bucket + N_BUCKETS - 1) % N_BUCKETS
                } else if db == 1 {
                    qi_bucket
                } else {
                    (qi_bucket + 1) % N_BUCKETS
                };

                for &ki in &bucket_positions[target_bucket] {
                    if ki > qi { continue; }
                    let delta = phases[qi] - phases[ki];
                    scores[ki] = (harmonic_n * delta).cos();
                }
            }

            if qi > 0 && scores[qi - 1] == f32::NEG_INFINITY {
                let delta = phases[qi] - phases[qi - 1];
                scores[qi - 1] = (harmonic_n * delta).cos();
            }
            if scores[qi] == f32::NEG_INFINITY {
                scores[qi] = 1.0;
            }

            // Softmax
            let max_s = scores[..=qi].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for ki in 0..=qi {
                if scores[ki] > f32::NEG_INFINITY {
                    scores[ki] = (scores[ki] - max_s).exp();
                    exp_sum += scores[ki];
                } else {
                    scores[ki] = 0.0;
                }
            }
            if exp_sum > 0.0 {
                for ki in 0..=qi { scores[ki] /= exp_sum; }
            }

            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for ki in 0..=qi {
                    if scores[ki] > 0.0 { sum += scores[ki] * v_all[ki][d]; }
                }
                out[qi][offset + d] = sum;
            }
        }
    }

    // Output projection
    out.iter().map(|o| linear(&weights.out_proj_w, &weights.out_proj_b, o)).collect()
}

// ─── Kerr-ODE derivative ──────────────────────────────────────

fn kerr_derivative(
    r: &[f32], s: &[f32],
    gamma: &[f32], omega: &[f32],
    alpha: f32, beta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = r.len();
    let mut dr = vec![0.0f32; n];
    let mut ds = vec![0.0f32; n];
    for k in 0..n {
        let mag_sq = r[k] * r[k] + s[k] * s[k];
        let mut ns = 0.0f32;
        if k >= 2 { ns += r[k-2]*r[k-2] + s[k-2]*s[k-2]; }
        if k >= 1 { ns += r[k-1]*r[k-1] + s[k-1]*s[k-1]; }
        if k+1 < n { ns += r[k+1]*r[k+1] + s[k+1]*s[k+1]; }
        if k+2 < n { ns += r[k+2]*r[k+2] + s[k+2]*s[k+2]; }
        let phi = omega[k] + alpha * mag_sq + beta * ns;
        dr[k] = -gamma[k] * r[k] - phi * s[k];
        ds[k] = -gamma[k] * s[k] + phi * r[k];
    }
    (dr, ds)
}

// ─── Dual-maestro Kerr-ODE FFN ──────────────────────────────

fn maestro_forward(weights: &MaestroWeights, x: &[f32]) -> Vec<f32> {
    let squeezed = linear(&weights.squeeze.w, &weights.squeeze.b, x);
    let activated: Vec<f32> = squeezed.iter().map(|&v| gelu(v)).collect();
    linear(&weights.process_1.w, &weights.process_1.b, &activated)
}

/// Perturbative ODE — single-pass analytical Kerr computation.
/// Matches wave-engine's perturbative implementation for inference consistency.
fn kerr_ode_forward(weights: &KerrWeights, x: &[f32]) -> Vec<f32> {
    let n_bands = weights.gamma_raw.len();
    let n_embd = n_bands * 2;

    let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus(g)).collect();

    let mut r_lin = vec![0.0f32; n_bands];
    let mut s_lin = vec![0.0f32; n_bands];
    for k in 0..n_bands {
        let r = x[k * 2];
        let s = x[k * 2 + 1];
        let decay = (-gamma[k]).exp();
        let cos_w = weights.omega[k].cos();
        let sin_w = weights.omega[k].sin();
        r_lin[k] = decay * (r * cos_w - s * sin_w);
        s_lin[k] = decay * (r * sin_w + s * cos_w);
    }

    let mut out = vec![0.0f32; n_embd];
    for k in 0..n_bands {
        let mag_sq = r_lin[k] * r_lin[k] + s_lin[k] * s_lin[k];
        let mut ns = 0.0f32;
        if k >= 2 { ns += r_lin[k-2]*r_lin[k-2] + s_lin[k-2]*s_lin[k-2]; }
        if k >= 1 { ns += r_lin[k-1]*r_lin[k-1] + s_lin[k-1]*s_lin[k-1]; }
        if k+1 < n_bands { ns += r_lin[k+1]*r_lin[k+1] + s_lin[k+1]*s_lin[k+1]; }
        if k+2 < n_bands { ns += r_lin[k+2]*r_lin[k+2] + s_lin[k+2]*s_lin[k+2]; }
        let delta_phi = weights.alpha * mag_sq + weights.beta * ns;
        out[k * 2]     = r_lin[k] - delta_phi * s_lin[k];
        out[k * 2 + 1] = s_lin[k] + delta_phi * r_lin[k];
    }
    out
}

fn dual_maestro_ffn_forward(weights: &KerrDualMaestroWeights, x: &[f32]) -> Vec<f32> {
    dual_maestro_ffn_forward_with_memory(weights, x, None)
}

/// FFN forward with optional memory injection into ODE initial conditions.
/// Returns the FFN output. If you also need the ODE output for memory extraction,
/// use dual_maestro_ffn_forward_extract() instead.
fn dual_maestro_ffn_forward_with_memory(
    weights: &KerrDualMaestroWeights,
    x: &[f32],
    memory_offset: Option<(&[f32], &[f32])>, // (r_offset, s_offset) per band
) -> Vec<f32> {
    let n_embd = x.len();
    let n_bands = n_embd / 2;

    // Maestro in (pre-ODE regulator)
    let mae_in = maestro_forward(&weights.maestro_in, x);
    let mut precond = vec![0.0f32; n_embd];
    for i in 0..n_embd { precond[i] = x[i] + mae_in[i]; }

    // Memory injection: add offsets to ODE initial conditions
    if let Some((r_off, s_off)) = memory_offset {
        for k in 0..n_bands {
            precond[k * 2] += r_off[k];
            precond[k * 2 + 1] += s_off[k];
        }
    }

    // AGC knee compression — must match wave-engine/src/common/ffn.rs
    agc_process(&mut precond, n_bands);

    // Kerr ODE
    let kerr_out = kerr_ode_forward(&weights.kerr, &precond);

    // Maestro out (post-ODE regulator)
    let mae_out = maestro_forward(&weights.maestro_out, &kerr_out);
    let mut regulated = vec![0.0f32; n_embd];
    for i in 0..n_embd { regulated[i] = kerr_out[i] + mae_out[i]; }

    // Out projection
    block_diagonal_forward(&weights.out_proj, &regulated)
}

/// FFN forward that also returns ODE output for memory state extraction (Fix 2).
fn dual_maestro_ffn_forward_extract(
    weights: &KerrDualMaestroWeights,
    x: &[f32],
    memory_offset: Option<(&[f32], &[f32])>,
) -> (Vec<f32>, Vec<f32>) {
    let n_embd = x.len();
    let n_bands = n_embd / 2;

    let mae_in = maestro_forward(&weights.maestro_in, x);
    let mut precond = vec![0.0f32; n_embd];
    for i in 0..n_embd { precond[i] = x[i] + mae_in[i]; }

    if let Some((r_off, s_off)) = memory_offset {
        for k in 0..n_bands {
            precond[k * 2] += r_off[k];
            precond[k * 2 + 1] += s_off[k];
        }
    }

    // AGC knee compression — must match wave-engine/src/common/ffn.rs
    agc_process(&mut precond, n_bands);

    let kerr_out = kerr_ode_forward(&weights.kerr, &precond);
    let mae_out = maestro_forward(&weights.maestro_out, &kerr_out);
    let mut regulated = vec![0.0f32; n_embd];
    for i in 0..n_embd { regulated[i] = kerr_out[i] + mae_out[i]; }

    let ffn_out = block_diagonal_forward(&weights.out_proj, &regulated);
    (ffn_out, kerr_out) // return ODE OUTPUT for memory extraction
}

// ─── Full forward pass ──────────────────────────────────────

impl ModelWeights {
    /// Forward pass: tokens → logits for all positions.
    pub fn forward(&self, tokens: &[usize]) -> Vec<Vec<f32>> {
        let t = tokens.len();
        let n_embd = self.config.n_embd();
        let n_bands = self.config.n_bands;

        // Embed: harmonic table lookup + positional encoding
        let mut hidden: Vec<Vec<f32>> = tokens.iter().enumerate().map(|(pos, &tok)| {
            let mut h = vec![0.0f32; n_embd];
            let tok_idx = tok.min(self.vocab_size - 1);
            let pos_idx = pos.min(self.config.block_size - 1);
            for i in 0..n_embd {
                h[i] = self.wte[tok_idx][i] + self.wpe[pos_idx][i];
            }
            h
        }).collect();

        // Parallel blocks: x = x + attn(LN(x)) + FFN(LN(x))
        for block in &self.blocks {
            let normed: Vec<Vec<f32>> = hidden.iter()
                .map(|h| layer_norm(h, &block.ln.weight, &block.ln.bias))
                .collect();

            // Attention path (harmonic coherence)
            let attn_out = wave_attention_forward(&block.attn, &normed, n_bands);

            // FFN path (dual-maestro Kerr-ODE) — same normed input
            let ffn_out: Vec<Vec<f32>> = normed.iter()
                .map(|x| dual_maestro_ffn_forward(&block.ffn, x))
                .collect();

            // Parallel residual
            hidden = (0..t).map(|i| {
                let mut v = vec![0.0f32; n_embd];
                for j in 0..n_embd {
                    v[j] = hidden[i][j] + attn_out[i][j] + ffn_out[i][j];
                }
                v
            }).collect();
        }

        // Final layer norm
        let post_ln: Vec<Vec<f32>> = hidden.iter()
            .map(|h| layer_norm(h, &self.ln_f.weight, &self.ln_f.bias))
            .collect();

        // LM head: project to vocab
        post_ln.iter().map(|h| {
            let mut logits = vec![0.0f32; self.vocab_size];
            for v in 0..self.vocab_size {
                for j in 0..n_embd {
                    logits[v] += self.lm_head[v][j] * h[j];
                }
            }
            logits
        }).collect()
    }

    /// Forward with wave memory injection (offsets added to ODE initial conditions).
    /// memory_offsets: one (r_offset, s_offset) per ODE layer.
    pub fn forward_with_memory(
        &self,
        tokens: &[usize],
        memory_offsets: Option<&[(&[f32], &[f32])]>,
    ) -> Vec<Vec<f32>> {
        let t = tokens.len();
        let n_embd = self.config.n_embd();
        let n_bands = self.config.n_bands;

        let mut hidden: Vec<Vec<f32>> = tokens.iter().enumerate().map(|(pos, &tok)| {
            let mut h = vec![0.0f32; n_embd];
            let tok_idx = tok.min(self.vocab_size - 1);
            let pos_idx = pos.min(self.config.block_size - 1);
            for i in 0..n_embd { h[i] = self.wte[tok_idx][i] + self.wpe[pos_idx][i]; }
            h
        }).collect();

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let normed: Vec<Vec<f32>> = hidden.iter()
                .map(|h| layer_norm(h, &block.ln.weight, &block.ln.bias))
                .collect();

            let attn_out = wave_attention_forward(&block.attn, &normed, n_bands);

            // Get memory offset for this layer (if available)
            let mem = memory_offsets.and_then(|m| m.get(layer_idx).copied());

            let ffn_out: Vec<Vec<f32>> = normed.iter()
                .map(|x| dual_maestro_ffn_forward_with_memory(&block.ffn, x, mem))
                .collect();

            hidden = (0..t).map(|i| {
                let mut v = vec![0.0f32; n_embd];
                for j in 0..n_embd { v[j] = hidden[i][j] + attn_out[i][j] + ffn_out[i][j]; }
                v
            }).collect();
        }

        let post_ln: Vec<Vec<f32>> = hidden.iter()
            .map(|h| layer_norm(h, &self.ln_f.weight, &self.ln_f.bias))
            .collect();

        post_ln.iter().map(|h| {
            let mut logits = vec![0.0f32; self.vocab_size];
            for v in 0..self.vocab_size {
                for j in 0..n_embd { logits[v] += self.lm_head[v][j] * h[j]; }
            }
            logits
        }).collect()
    }

    /// Prefill the KV-cache with prompt tokens (full forward, caches all positions).
    /// memory_offsets: one (r_offset, s_offset) per ODE layer, or None.
    pub fn prefill(&self, tokens: &[usize], cache: &mut KvCache, memory_offsets: Option<&[(&[f32], &[f32])]>) -> Vec<f32> {
        cache.clear();
        let t = tokens.len();
        let n_embd = self.config.n_embd();
        let n_bands = self.config.n_bands;
        let n_head = self.config.n_head;
        let head_dim = n_embd / n_head;

        let mut hidden: Vec<Vec<f32>> = tokens.iter().enumerate().map(|(pos, &tok)| {
            let mut h = vec![0.0f32; n_embd];
            let tok_idx = tok.min(self.vocab_size - 1);
            let pos_idx = pos.min(self.config.block_size - 1);
            for i in 0..n_embd { h[i] = self.wte[tok_idx][i] + self.wpe[pos_idx][i]; }
            h
        }).collect();

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let normed: Vec<Vec<f32>> = hidden.iter()
                .map(|h| layer_norm(h, &block.ln.weight, &block.ln.bias))
                .collect();

            // Cache phases and values for all positions
            let kv = &mut cache.layers[layer_idx];
            for head in 0..n_head {
                let offset = head * head_dim;
                for pos in 0..t {
                    let phase = project_phase(&normed[pos],
                        &block.attn.heads[head].phase_proj_w,
                        &block.attn.heads[head].phase_proj_b);
                    kv.phases[head].push(phase);

                    let normalized = ((phase % std::f32::consts::TAU) + std::f32::consts::TAU) % std::f32::consts::TAU;
                    let bucket = ((normalized / (std::f32::consts::TAU / 8.0)) as usize).min(7);
                    kv.buckets[head].push(bucket);

                    let mut v = vec![0.0f32; head_dim];
                    for d in 0..head_dim {
                        let mut sum = 0.0f32;
                        for j in 0..head_dim {
                            sum += block.attn.heads[head].v_proj_w[d][j] * normed[pos][offset + j];
                        }
                        v[d] = sum + block.attn.heads[head].v_proj_b[d];
                    }
                    kv.values[head].push(v);
                }
            }

            // Full attention for all positions (same as uncached)
            let attn_out = wave_attention_forward(&block.attn, &normed, n_bands);

            // FFN for all positions (with memory injection if available)
            let mem = memory_offsets.and_then(|m| m.get(layer_idx).copied());
            let ffn_out: Vec<Vec<f32>> = normed.iter()
                .map(|x| dual_maestro_ffn_forward_with_memory(&block.ffn, x, mem))
                .collect();

            hidden = (0..t).map(|i| {
                let mut v = vec![0.0f32; n_embd];
                for j in 0..n_embd { v[j] = hidden[i][j] + attn_out[i][j] + ffn_out[i][j]; }
                v
            }).collect();

            // Cache hidden states for this layer (used if we need ODE states)
            cache.hidden_per_layer[layer_idx] = hidden.clone();
        }

        cache.n_cached = t;

        // Return last position's logits
        let last = layer_norm(&hidden[t - 1], &self.ln_f.weight, &self.ln_f.bias);
        let mut logits = vec![0.0f32; self.vocab_size];
        for v in 0..self.vocab_size {
            for j in 0..n_embd { logits[v] += self.lm_head[v][j] * last[j]; }
        }
        logits
    }

    /// Generate one token using KV-cache. O(1) per layer instead of O(T).
    /// Only processes the new position against cached phases/values.
    pub fn forward_one_cached(&self, token: usize, pos: usize, cache: &mut KvCache, memory_offsets: Option<&[(&[f32], &[f32])]>) -> Vec<f32> {
        let n_embd = self.config.n_embd();
        let n_bands = self.config.n_bands;
        let n_head = self.config.n_head;
        let head_dim = n_embd / n_head;
        let pos_idx = pos.min(self.config.block_size - 1);

        // Embed new token
        let tok_idx = token.min(self.vocab_size - 1);
        let mut h = vec![0.0f32; n_embd];
        for i in 0..n_embd { h[i] = self.wte[tok_idx][i] + self.wpe[pos_idx][i]; }

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let normed = layer_norm(&h, &block.ln.weight, &block.ln.bias);

            // ─── Attention: score new position against ALL cached positions ───
            let kv = &mut cache.layers[layer_idx];
            let n_cached = kv.phases[0].len();
            let mut attn_out = vec![0.0f32; n_embd];

            for head in 0..n_head {
                let harmonic_n = softplus(block.attn.heads[head].harmonic_raw);
                let offset = head * head_dim;

                // New position's phase and value
                let new_phase = project_phase(&normed,
                    &block.attn.heads[head].phase_proj_w,
                    &block.attn.heads[head].phase_proj_b);

                let mut new_v = vec![0.0f32; head_dim];
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for j in 0..head_dim {
                        sum += block.attn.heads[head].v_proj_w[d][j] * normed[offset + j];
                    }
                    new_v[d] = sum + block.attn.heads[head].v_proj_b[d];
                }

                // Score against all cached positions + self
                let total = n_cached + 1; // cached + new
                let mut scores = vec![0.0f32; total];

                // Sparse: only score same + adjacent buckets
                let new_normalized = ((new_phase % std::f32::consts::TAU) + std::f32::consts::TAU) % std::f32::consts::TAU;
                let new_bucket = ((new_normalized / (std::f32::consts::TAU / 8.0)) as usize).min(7);

                for ki in 0..n_cached {
                    let ki_bucket = kv.buckets[head][ki];
                    let bucket_diff = (new_bucket as isize - ki_bucket as isize).unsigned_abs();
                    let adjacent = bucket_diff <= 1 || bucket_diff >= 7; // wrapping
                    if adjacent || ki + 1 >= n_cached { // always attend to last cached
                        let delta = new_phase - kv.phases[head][ki];
                        scores[ki] = (harmonic_n * delta).cos();
                    } else {
                        scores[ki] = f32::NEG_INFINITY;
                    }
                }
                // Self-attention
                scores[n_cached] = 1.0;

                // Softmax
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                for s in &mut scores {
                    if *s > f32::NEG_INFINITY {
                        *s = (*s - max_s).exp();
                        exp_sum += *s;
                    } else {
                        *s = 0.0;
                    }
                }
                if exp_sum > 0.0 { for s in &mut scores { *s /= exp_sum; } }

                // Weighted sum of cached values + self
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for ki in 0..n_cached {
                        if scores[ki] > 0.0 { sum += scores[ki] * kv.values[head][ki][d]; }
                    }
                    sum += scores[n_cached] * new_v[d]; // self
                    attn_out[offset + d] = sum;
                }

                // Cache new position
                kv.phases[head].push(new_phase);
                kv.values[head].push(new_v);
                kv.buckets[head].push(new_bucket);
            }

            // Output projection for attention
            let attn_projected = linear(&block.attn.out_proj_w, &block.attn.out_proj_b, &attn_out);

            // ─── FFN: process just this one token (with memory if available) ───
            let mem = memory_offsets.and_then(|m| m.get(layer_idx).copied());
            let ffn_out = dual_maestro_ffn_forward_with_memory(&block.ffn, &normed, mem);

            // Parallel residual
            for j in 0..n_embd { h[j] = h[j] + attn_projected[j] + ffn_out[j]; }
        }

        cache.n_cached += 1;

        // LM head
        let post_ln = layer_norm(&h, &self.ln_f.weight, &self.ln_f.bias);
        let mut logits = vec![0.0f32; self.vocab_size];
        for v in 0..self.vocab_size {
            for j in 0..n_embd { logits[v] += self.lm_head[v][j] * post_ln[j]; }
        }
        logits
    }

    /// Extract ODE OUTPUT states for memory accumulation (Fix 2: captures ODE output, not input).
    /// Full forward pass — use extract_ode_states_from_cache() when KV-cache is available.
    pub fn extract_ode_states(&self, tokens: &[usize]) -> Vec<(Vec<f32>, Vec<f32>)> {
        let n_embd = self.config.n_embd();
        let n_bands = self.config.n_bands;
        let t = tokens.len();

        let mut hidden: Vec<Vec<f32>> = tokens.iter().enumerate().map(|(pos, &tok)| {
            let mut h = vec![0.0f32; n_embd];
            let tok_idx = tok.min(self.vocab_size - 1);
            let pos_idx = pos.min(self.config.block_size - 1);
            for i in 0..n_embd { h[i] = self.wte[tok_idx][i] + self.wpe[pos_idx][i]; }
            h
        }).collect();

        let mut ode_states = Vec::new();

        for block in &self.blocks {
            let normed: Vec<Vec<f32>> = hidden.iter()
                .map(|h| layer_norm(h, &block.ln.weight, &block.ln.bias))
                .collect();

            let attn_out = wave_attention_forward(&block.attn, &normed, n_bands);

            // FFN with ODE output extraction — captures post-RK4 dynamics
            let mut avg_r = vec![0.0f32; n_bands];
            let mut avg_s = vec![0.0f32; n_bands];
            let mut ffn_out = Vec::with_capacity(t);
            for pos in 0..t {
                let (fout, ode_out) = dual_maestro_ffn_forward_extract(&block.ffn, &normed[pos], None);
                // Accumulate ODE OUTPUT (the transformed representation)
                for k in 0..n_bands {
                    avg_r[k] += ode_out[k * 2];
                    avg_s[k] += ode_out[k * 2 + 1];
                }
                ffn_out.push(fout);
            }
            let scale = 1.0 / t as f32;
            for k in 0..n_bands { avg_r[k] *= scale; avg_s[k] *= scale; }
            ode_states.push((avg_r, avg_s));

            hidden = (0..t).map(|i| {
                let mut v = vec![0.0f32; n_embd];
                for j in 0..n_embd { v[j] = hidden[i][j] + attn_out[i][j] + ffn_out[i][j]; }
                v
            }).collect();
        }

        ode_states
    }

    /// Extract ODE states from KV-cache (Fix 3: no duplicate forward pass).
    /// Uses last position's hidden state from each layer. Runs only the FFN
    /// (maestro_in + ODE) per layer — not the full model.
    pub fn extract_ode_states_from_cache(&self, cache: &KvCache) -> Vec<(Vec<f32>, Vec<f32>)> {
        let n_embd = self.config.n_embd();
        let n_bands = self.config.n_bands;
        let mut ode_states = Vec::new();

        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let hidden = &cache.hidden_per_layer[layer_idx];
            if hidden.is_empty() {
                ode_states.push((vec![0.0; n_bands], vec![0.0; n_bands]));
                continue;
            }
            // Use last position (most recent context)
            let last = &hidden[hidden.len() - 1];
            let normed = layer_norm(last, &block.ln.weight, &block.ln.bias);
            let (_ffn_out, ode_out) = dual_maestro_ffn_forward_extract(&block.ffn, &normed, None);

            let mut r = vec![0.0f32; n_bands];
            let mut s = vec![0.0f32; n_bands];
            for k in 0..n_bands {
                r[k] = ode_out[k * 2];
                s[k] = ode_out[k * 2 + 1];
            }
            ode_states.push((r, s));
        }

        ode_states
    }
}
