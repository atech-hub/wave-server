//! Full wave-engine model — CPU reference implementation.
//!
//! Matches phaseC_integrated.py exactly for inference.
//! Architecture: 4 blocks, each with CausalSelfAttention + FFN.
//!   Block 0: Attention + PerBandLinear
//!   Blocks 1-3: Attention + KerrMaestroAdd (Kerr-ODE + Maestro)

use std::f32::consts::PI;

// ─── Model configuration ──────────────────────────────────────

/// Runtime-configurable architecture dimensions.
/// Stored in ModelWeights so every function can derive dims from data or config.
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
    /// Default config matching the original 128-dim architecture.
    pub fn default_128() -> Self {
        Self {
            n_bands: 64,
            n_head: 4,
            n_layers: 4,
            maestro_dim: 16,
            block_size: 256,
            rk4_n_steps: 8,
        }
    }

    pub fn n_embd(&self) -> usize { self.n_bands * 2 }
    #[allow(dead_code)]
    pub fn head_dim(&self) -> usize { self.n_embd() / self.n_head }
    #[allow(dead_code)]
    pub fn rk4_dt(&self) -> f32 { 1.0 / self.rk4_n_steps as f32 }

    pub fn validate(&self) {
        assert!(self.n_bands > 0, "n_bands must be > 0");
        assert!(self.n_head > 0, "n_head must be > 0");
        assert_eq!(self.n_embd() % self.n_head, 0, "n_embd must be divisible by n_head");
        assert!(self.rk4_n_steps > 0, "rk4_n_steps must be > 0");
    }
}

// Legacy compile-time constants — kept for init.rs defaults and backward compatibility.
// New code should use ModelConfig or derive from data dimensions.
pub const N_BANDS: usize = 64;
pub const N_EMBD: usize = 128;
pub const N_HEAD: usize = 4;
#[allow(dead_code)]
pub const HEAD_DIM: usize = N_EMBD / N_HEAD;
pub const BLOCK_SIZE: usize = 256;
pub const MAESTRO_DIM: usize = 16;
pub const RK4_N_STEPS: usize = 8;
#[allow(dead_code)]
pub const RK4_DT: f32 = 1.0 / RK4_N_STEPS as f32;
pub const N_LAYERS: usize = 4;

/// Softplus: log(1 + exp(x))
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x  // avoid overflow
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// GELU activation (approximate version matching PyTorch default)
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

/// Build frozen harmonic embedding table.
/// Returns [vocab_size][n_embd] array.
#[allow(dead_code)]
pub fn build_harmonic_table(vocab_size: usize) -> Vec<Vec<f32>> {
    // n_embd derived from vocab table: caller sizes it via config, but the formula
    // only depends on vocab_size and n_bands. We use N_EMBD for the legacy path
    // and data-derived for the config path. Since this is called from init_model
    // which knows config, the table size is correct.
    let n_embd = N_EMBD; // Legacy — will be parameterized in Phase 2
    let nh = n_embd / 2;
    let scale = 1.0 / (nh as f32).sqrt();
    let mut table = vec![vec![0.0f32; n_embd]; vocab_size];

    for c in 0..vocab_size {
        let theta = c as f32 * 2.0 * PI / vocab_size as f32;
        for h in 0..nh {
            let angle = (h + 1) as f32 * theta;
            table[c][h * 2] = angle.cos() * scale;
            table[c][h * 2 + 1] = angle.sin() * scale;
        }
    }
    table
}

/// Build frozen harmonic embedding table with explicit n_embd.
pub fn build_harmonic_table_sized(vocab_size: usize, n_embd: usize) -> Vec<Vec<f32>> {
    let nh = n_embd / 2;
    let scale = 1.0 / (nh as f32).sqrt();
    let mut table = vec![vec![0.0f32; n_embd]; vocab_size];

    for c in 0..vocab_size {
        let theta = c as f32 * 2.0 * PI / vocab_size as f32;
        for h in 0..nh {
            let angle = (h + 1) as f32 * theta;
            table[c][h * 2] = angle.cos() * scale;
            table[c][h * 2 + 1] = angle.sin() * scale;
        }
    }
    table
}

/// Build positional encoding table.
/// Returns [block_size][n_embd] array.
pub fn build_positional_table(block_size: usize, n_embd: usize) -> Vec<Vec<f32>> {
    let nh = n_embd / 2;
    let scale = 1.0 / (nh as f32).sqrt();
    let mut table = vec![vec![0.0f32; n_embd]; block_size];

    for pos in 0..block_size {
        for h in 0..nh {
            let freq = 1.0 / 10000.0_f32.powf(2.0 * h as f32 / n_embd as f32);
            table[pos][h * 2] = (pos as f32 * freq).cos() * scale;
            table[pos][h * 2 + 1] = (pos as f32 * freq).sin() * scale;
        }
    }
    table
}

// ─── Linear algebra helpers ──────────────────────────────────────

/// Matrix-vector multiply: y = W @ x + b
/// W is [out_dim][in_dim], x is [in_dim], b is [out_dim]
#[inline]
fn linear(w: &[Vec<f32>], b: &[f32], x: &[f32]) -> Vec<f32> {
    w.iter()
        .zip(b.iter())
        .map(|(row, &bias)| {
            bias + row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum::<f32>()
        })
        .collect()
}

/// Layer normalization: y = (x - mean) / sqrt(var + eps) * weight + bias
fn layer_norm(x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let std = (var + 1e-5).sqrt();

    let mut y = vec![0.0f32; n];
    for i in 0..n {
        y[i] = (x[i] - mean) / std * weight[i] + bias[i];
    }
    y
}

// ─── Weight structures ──────────────────────────────────────────

/// Weights for a Linear layer.
#[derive(Clone)]
pub struct LinearWeights {
    pub w: Vec<Vec<f32>>,  // [out_dim][in_dim]
    pub b: Vec<f32>,       // [out_dim]
}

/// Weights for LayerNorm.
#[derive(Clone)]
pub struct LayerNormWeights {
    pub weight: Vec<f32>,  // [dim]
    pub bias: Vec<f32>,    // [dim]
}

/// Weights for CausalSelfAttention.
#[derive(Clone)]
pub struct AttentionWeights {
    pub c_attn: LinearWeights,  // [3*N_EMBD, N_EMBD]
    pub c_proj: LinearWeights,  // [N_EMBD, N_EMBD]
    pub n_head: usize,          // Number of attention heads (for head_dim derivation)
}

/// Weights for PerBandLinear (Block 0 FFN).
#[derive(Clone)]
pub struct PerBandLinearWeights {
    pub band_w: Vec<[[f32; 2]; 2]>,  // [N_BANDS][2][2]
    pub band_b: Vec<[f32; 2]>,       // [N_BANDS][2]
    pub out_proj: LinearWeights,
}

/// Weights for Kerr-ODE layer.
#[derive(Clone)]
pub struct KerrWeights {
    pub gamma_raw: Vec<f32>,  // [N_BANDS] (before softplus)
    pub omega: Vec<f32>,      // [N_BANDS]
    pub alpha: f32,
    pub beta: f32,
    pub rk4_n_steps: usize,   // ODE integration steps (default 8)
}

/// Weights for Maestro.
#[derive(Clone)]
pub struct MaestroWeights {
    pub squeeze: LinearWeights,   // [MAESTRO_DIM, N_EMBD]
    pub process_1: LinearWeights, // [N_EMBD, MAESTRO_DIM]
}

/// Weights for KerrMaestroAdd block (Blocks 1-3 FFN).
#[derive(Clone)]
pub struct KerrMaestroAddWeights {
    pub kerr: KerrWeights,
    pub maestro: MaestroWeights,
    pub out_proj: LinearWeights,
}

/// Weights for one Block.
#[derive(Clone)]
pub struct BlockWeights {
    pub ln_1: LayerNormWeights,
    pub attn: AttentionWeights,
    pub ln_2: LayerNormWeights,
    pub ffn: FfnWeights,
}

/// FFN can be PerBandLinear (block 0) or KerrMaestroAdd (blocks 1-3).
#[derive(Clone)]
pub enum FfnWeights {
    PerBand(PerBandLinearWeights),
    KerrMaestro(KerrMaestroAddWeights),
}

/// Full model weights.
pub struct ModelWeights {
    pub config: ModelConfig,
    pub vocab_size: usize,
    pub wte_phase: Vec<Vec<f32>>,  // [vocab_size][N_EMBD] (frozen)
    pub wpe: Vec<Vec<f32>>,        // [BLOCK_SIZE][N_EMBD] (frozen)
    pub blocks: Vec<BlockWeights>, // n_layers blocks
    pub ln_f: LayerNormWeights,
    pub lm_head: Vec<Vec<f32>>,    // [vocab_size][N_EMBD] (no bias)
}

// ─── Forward pass ───────────────────────────────────────────────

impl ModelWeights {
    /// Full forward pass: token indices → logits.
    pub fn forward(&self, tokens: &[usize]) -> Vec<Vec<f32>> {
        self.forward_with_memory(tokens, None)
    }

    /// Forward pass with optional wave memory injection.
    ///
    /// `memory_offsets` is a slice of per-layer (r_offset, s_offset) pairs.
    /// Each offset is added to the Kerr-ODE initial conditions before RK4.
    /// When None, the code path is identical to `forward()` (bit-identical).
    pub fn forward_with_memory(
        &self,
        tokens: &[usize],
        memory_offsets: Option<&[(&[f32], &[f32])]>,
    ) -> Vec<Vec<f32>> {
        let t = tokens.len();
        let n_embd = self.config.n_embd();
        assert!(t <= self.config.block_size);

        // Embedding + positional encoding
        let mut hidden: Vec<Vec<f32>> = Vec::with_capacity(t);
        for (pos, &tok) in tokens.iter().enumerate() {
            let mut h = vec![0.0f32; n_embd];
            for i in 0..n_embd {
                h[i] = self.wte_phase[tok][i] + self.wpe[pos][i];
            }
            hidden.push(h);
        }

        // Process through blocks — track ODE layer index for memory injection
        let mut ode_layer = 0usize;
        for block in &self.blocks {
            let mem = match (&block.ffn, memory_offsets) {
                (FfnWeights::KerrMaestro(_), Some(offsets)) if ode_layer < offsets.len() => {
                    let m = Some(offsets[ode_layer]);
                    ode_layer += 1;
                    m
                }
                (FfnWeights::KerrMaestro(_), _) => { ode_layer += 1; None }
                _ => None, // PerBandLinear — no ODE, no memory
            };
            hidden = self.forward_block(block, &hidden, mem);
        }

        // Final layer norm + LM head
        let mut logits = Vec::with_capacity(t);
        for h in &hidden {
            let normed = layer_norm(h, &self.ln_f.weight, &self.ln_f.bias);
            let l = linear_no_bias(&self.lm_head, &normed);
            logits.push(l);
        }

        logits
    }

    fn forward_block(
        &self,
        block: &BlockWeights,
        hidden: &[Vec<f32>],
        memory: Option<(&[f32], &[f32])>,
    ) -> Vec<Vec<f32>> {
        let t = hidden.len();
        let n_embd = self.config.n_embd();

        // x = x + attn(ln_1(x))
        let normed_1: Vec<Vec<f32>> = hidden.iter()
            .map(|h| layer_norm(h, &block.ln_1.weight, &block.ln_1.bias))
            .collect();
        let attn_out = self.causal_self_attention(&block.attn, &normed_1);
        let mut h: Vec<Vec<f32>> = (0..t)
            .map(|i| {
                let mut v = vec![0.0f32; n_embd];
                for j in 0..n_embd { v[j] = hidden[i][j] + attn_out[i][j]; }
                v
            })
            .collect();

        // x = x + ffn(ln_2(x))
        let normed_2: Vec<Vec<f32>> = h.iter()
            .map(|x| layer_norm(x, &block.ln_2.weight, &block.ln_2.bias))
            .collect();
        let ffn_out = match &block.ffn {
            FfnWeights::PerBand(w) => self.per_band_linear(w, &normed_2),
            FfnWeights::KerrMaestro(w) => self.kerr_maestro_add_with_memory(w, &normed_2, memory),
        };
        for i in 0..t {
            for j in 0..n_embd { h[i][j] += ffn_out[i][j]; }
        }

        h
    }

    fn causal_self_attention(&self, weights: &AttentionWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let t = x.len();
        let n_embd = self.config.n_embd();
        let n_head = weights.n_head;
        let head_dim = n_embd / n_head;

        // Compute Q, K, V for all positions
        let mut q_all = vec![vec![0.0f32; n_embd]; t];
        let mut k_all = vec![vec![0.0f32; n_embd]; t];
        let mut v_all = vec![vec![0.0f32; n_embd]; t];

        for pos in 0..t {
            let qkv = linear(&weights.c_attn.w, &weights.c_attn.b, &x[pos]);
            for i in 0..n_embd {
                q_all[pos][i] = qkv[i];
                k_all[pos][i] = qkv[n_embd + i];
                v_all[pos][i] = qkv[2 * n_embd + i];
            }
        }

        // Multi-head attention
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut out = vec![vec![0.0f32; n_embd]; t];

        for head in 0..n_head {
            let offset = head * head_dim;

            // Compute attention scores for this head
            for qi in 0..t {
                // Compute attention weights
                let mut att = vec![f32::NEG_INFINITY; t];
                for ki in 0..=qi {  // causal: only attend to past
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_all[qi][offset + d] * k_all[ki][offset + d];
                    }
                    att[ki] = dot * scale;
                }

                // Softmax
                let max_att = att[..=qi].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                for ki in 0..=qi {
                    att[ki] = (att[ki] - max_att).exp();
                    exp_sum += att[ki];
                }
                for ki in 0..=qi {
                    att[ki] /= exp_sum;
                }

                // Weighted sum of values
                for d in 0..head_dim {
                    let mut sum = 0.0f32;
                    for ki in 0..=qi {
                        sum += att[ki] * v_all[ki][offset + d];
                    }
                    out[qi][offset + d] = sum;
                }
            }
        }

        // Output projection
        let result: Vec<Vec<f32>> = out.iter()
            .map(|o| linear(&weights.c_proj.w, &weights.c_proj.b, o))
            .collect();

        result
    }

    pub fn per_band_linear(&self, weights: &PerBandLinearWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let t = x.len();
        let n_bands = weights.band_w.len();
        let n_embd = n_bands * 2;
        let mut result = Vec::with_capacity(t);

        for pos in 0..t {
            let mut bands_out = vec![0.0f32; n_embd];

            for band in 0..n_bands {
                let r_in = x[pos][band * 2];
                let s_in = x[pos][band * 2 + 1];
                let w = &weights.band_w[band];
                let b = &weights.band_b[band];

                // y = W @ [r, s] + b  (2x2 matrix)
                bands_out[band * 2] = w[0][0] * r_in + w[1][0] * s_in + b[0];
                bands_out[band * 2 + 1] = w[0][1] * r_in + w[1][1] * s_in + b[1];
            }

            let projected = linear(&weights.out_proj.w, &weights.out_proj.b, &bands_out);
            result.push(projected);
        }

        result
    }

    pub fn kerr_maestro_add(&self, weights: &KerrMaestroAddWeights, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        self.kerr_maestro_add_with_memory(weights, x, None)
    }

    pub fn kerr_maestro_add_with_memory(
        &self,
        weights: &KerrMaestroAddWeights,
        x: &[Vec<f32>],
        memory: Option<(&[f32], &[f32])>,
    ) -> Vec<Vec<f32>> {
        let t = x.len();
        let mut result = Vec::with_capacity(t);

        for pos in 0..t {
            // Kerr path (with optional memory injection)
            let kerr_out = self.kerr_ode_forward_with_memory(&weights.kerr, &x[pos], memory);

            // Maestro path (no memory injection — global coordination only)
            let maestro_out = self.maestro_forward(&weights.maestro, &x[pos]);

            // Combine + project
            let n_embd = kerr_out.len();
            let mut combined = vec![0.0f32; n_embd];
            for i in 0..n_embd {
                combined[i] = kerr_out[i] + maestro_out[i];
            }

            let projected = linear(&weights.out_proj.w, &weights.out_proj.b, &combined);
            result.push(projected);
        }

        result
    }

    pub fn kerr_ode_forward(&self, weights: &KerrWeights, x: &[f32]) -> Vec<f32> {
        self.kerr_ode_forward_with_memory(weights, x, None)
    }

    /// Kerr-ODE forward pass with optional wave memory injection.
    ///
    /// When `memory` is Some((r_offsets, s_offsets)), the offsets are added
    /// to the initial conditions before RK4 integration. When None, the
    /// code path is identical to the original (bit-identical baseline).
    pub fn kerr_ode_forward_with_memory(
        &self,
        weights: &KerrWeights,
        x: &[f32],
        memory: Option<(&[f32], &[f32])>,
    ) -> Vec<f32> {
        let n_bands = weights.gamma_raw.len();
        let n_embd = n_bands * 2;
        let n_steps = weights.rk4_n_steps;
        let dt = 1.0 / n_steps as f32;

        // Split into real and imaginary parts
        let mut r = vec![0.0f32; n_bands];
        let mut s = vec![0.0f32; n_bands];
        for k in 0..n_bands {
            r[k] = x[k * 2];
            s[k] = x[k * 2 + 1];
        }

        // Wave memory injection: add offsets to initial conditions
        if let Some((r_mem, s_mem)) = memory {
            for k in 0..n_bands.min(r_mem.len()) {
                r[k] += r_mem[k];
                s[k] += s_mem[k];
            }
        }

        // Compute gamma (softplus of raw)
        let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus(g)).collect();

        // RK4 integration steps
        for _ in 0..n_steps {
            let (r_new, s_new) = rk4_step(&r, &s, dt, &gamma,
                                           &weights.omega, weights.alpha, weights.beta);
            r = r_new;
            s = s_new;
        }

        // Reinterleave
        let mut out = vec![0.0f32; n_embd];
        for k in 0..n_bands {
            out[k * 2] = r[k];
            out[k * 2 + 1] = s[k];
        }
        out
    }

    /// Extract final ODE states from all layers and positions.
    /// Returns ode_states[ode_layer] = (r_avg, s_avg) averaged across positions.
    /// Used for wave memory accumulation — run once per conversation.
    pub fn extract_ode_states(
        &self,
        tokens: &[usize],
        memory_offsets: Option<&[(&[f32], &[f32])]>,
    ) -> Vec<(Vec<f32>, Vec<f32>)> {
        let t = tokens.len();
        let n_embd = self.config.n_embd();
        let n_bands = self.config.n_bands;
        assert!(t <= self.config.block_size);

        // Embedding + positional
        let mut hidden: Vec<Vec<f32>> = Vec::with_capacity(t);
        for (pos, &tok) in tokens.iter().enumerate() {
            let mut h = vec![0.0f32; n_embd];
            for i in 0..n_embd { h[i] = self.wte_phase[tok][i] + self.wpe[pos][i]; }
            hidden.push(h);
        }

        let mut ode_states: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();
        let mut ode_layer = 0usize;

        for block in &self.blocks {
            // Attention + residual
            let normed_1: Vec<Vec<f32>> = hidden.iter()
                .map(|h| layer_norm(h, &block.ln_1.weight, &block.ln_1.bias))
                .collect();
            let attn_out = self.causal_self_attention(&block.attn, &normed_1);
            let mut h: Vec<Vec<f32>> = (0..t).map(|i| {
                let mut v = vec![0.0f32; n_embd];
                for j in 0..n_embd { v[j] = hidden[i][j] + attn_out[i][j]; }
                v
            }).collect();

            // FFN
            let normed_2: Vec<Vec<f32>> = h.iter()
                .map(|x| layer_norm(x, &block.ln_2.weight, &block.ln_2.bias))
                .collect();

            match &block.ffn {
                FfnWeights::PerBand(w) => {
                    let ffn_out = self.per_band_linear(w, &normed_2);
                    for i in 0..t {
                        for j in 0..n_embd { h[i][j] += ffn_out[i][j]; }
                    }
                }
                FfnWeights::KerrMaestro(w) => {
                    let mem = match memory_offsets {
                        Some(offsets) if ode_layer < offsets.len() => Some(offsets[ode_layer]),
                        _ => None,
                    };

                    // Extract ODE states from ALL positions, average them
                    let mut avg_r = vec![0.0f32; n_bands];
                    let mut avg_s = vec![0.0f32; n_bands];

                    for pos in 0..t {
                        // Run Kerr-ODE and capture final (r, s)
                        let x = &normed_2[pos];
                        let mut r = vec![0.0f32; n_bands];
                        let mut s = vec![0.0f32; n_bands];
                        for k in 0..n_bands {
                            r[k] = x[k * 2];
                            s[k] = x[k * 2 + 1];
                        }
                        if let Some((r_mem, s_mem)) = mem {
                            for k in 0..n_bands.min(r_mem.len()) {
                                r[k] += r_mem[k];
                                s[k] += s_mem[k];
                            }
                        }
                        let gamma: Vec<f32> = w.kerr.gamma_raw.iter()
                            .map(|&g| softplus(g)).collect();
                        let n_steps = w.kerr.rk4_n_steps;
                        let dt = 1.0 / n_steps as f32;
                        for _ in 0..n_steps {
                            let (r_new, s_new) = rk4_step(&r, &s, dt, &gamma,
                                &w.kerr.omega, w.kerr.alpha, w.kerr.beta);
                            r = r_new;
                            s = s_new;
                        }
                        for k in 0..n_bands {
                            avg_r[k] += r[k];
                            avg_s[k] += s[k];
                        }
                    }

                    // Average across positions
                    let scale = 1.0 / t as f32;
                    for k in 0..n_bands {
                        avg_r[k] *= scale;
                        avg_s[k] *= scale;
                    }
                    ode_states.push((avg_r, avg_s));

                    // Normal forward for hidden state propagation
                    let ffn_out = self.kerr_maestro_add_with_memory(w, &normed_2, mem);
                    for i in 0..t {
                        for j in 0..n_embd { h[i][j] += ffn_out[i][j]; }
                    }

                    ode_layer += 1;
                }
            }
            hidden = h;
        }

        ode_states
    }

    pub fn maestro_forward(&self, weights: &MaestroWeights, x: &[f32]) -> Vec<f32> {
        // Squeeze: 128 → 16
        let squeezed = linear(&weights.squeeze.w, &weights.squeeze.b, x);

        // GELU activation
        let activated: Vec<f32> = squeezed.iter().map(|&v| gelu(v)).collect();

        // Process: 16 → 128
        linear(&weights.process_1.w, &weights.process_1.b, &activated)
    }
}

// ─── Kerr-ODE derivative and RK4 ───────────────────────────────

fn kerr_derivative(
    r: &[f32], s: &[f32],
    gamma: &[f32], omega: &[f32],
    alpha: f32, beta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = r.len();
    let mut dr = vec![0.0f32; n];
    let mut ds = vec![0.0f32; n];

    // Compute mag_sq for all bands
    let mag_sq: Vec<f32> = (0..n).map(|k| r[k] * r[k] + s[k] * s[k]).collect();

    // Conv1d with kernel [1, 1, 0, 1, 1], padding=2
    let mut ns = vec![0.0f32; n];
    for k in 0..n {
        if k >= 2 { ns[k] += mag_sq[k - 2]; }
        if k >= 1 { ns[k] += mag_sq[k - 1]; }
        if k + 1 < n { ns[k] += mag_sq[k + 1]; }
        if k + 2 < n { ns[k] += mag_sq[k + 2]; }
    }

    for k in 0..n {
        let phi = omega[k] + alpha * mag_sq[k] + beta * ns[k];
        dr[k] = -gamma[k] * r[k] - phi * s[k];
        ds[k] = -gamma[k] * s[k] + phi * r[k];
    }

    (dr, ds)
}

fn rk4_step(
    r: &[f32], s: &[f32], dt: f32,
    gamma: &[f32], omega: &[f32],
    alpha: f32, beta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = r.len();

    // k1
    let (dr1, ds1) = kerr_derivative(r, s, gamma, omega, alpha, beta);

    // k2
    let r2: Vec<f32> = (0..n).map(|k| r[k] + 0.5 * dt * dr1[k]).collect();
    let s2: Vec<f32> = (0..n).map(|k| s[k] + 0.5 * dt * ds1[k]).collect();
    let (dr2, ds2) = kerr_derivative(&r2, &s2, gamma, omega, alpha, beta);

    // k3
    let r3: Vec<f32> = (0..n).map(|k| r[k] + 0.5 * dt * dr2[k]).collect();
    let s3: Vec<f32> = (0..n).map(|k| s[k] + 0.5 * dt * ds2[k]).collect();
    let (dr3, ds3) = kerr_derivative(&r3, &s3, gamma, omega, alpha, beta);

    // k4
    let r4: Vec<f32> = (0..n).map(|k| r[k] + dt * dr3[k]).collect();
    let s4: Vec<f32> = (0..n).map(|k| s[k] + dt * ds3[k]).collect();
    let (dr4, ds4) = kerr_derivative(&r4, &s4, gamma, omega, alpha, beta);

    // Combine: y_new = y + (dt/6)(k1 + 2k2 + 2k3 + k4)
    let dt6 = dt / 6.0;
    let r_new: Vec<f32> = (0..n)
        .map(|k| r[k] + dt6 * (dr1[k] + 2.0 * dr2[k] + 2.0 * dr3[k] + dr4[k]))
        .collect();
    let s_new: Vec<f32> = (0..n)
        .map(|k| s[k] + dt6 * (ds1[k] + 2.0 * ds2[k] + 2.0 * ds3[k] + ds4[k]))
        .collect();

    (r_new, s_new)
}

/// Public wrapper for rk4_step (needed by backward.rs).
pub fn rk4_step_public(
    r: &[f32], s: &[f32], dt: f32,
    gamma: &[f32], omega: &[f32],
    alpha: f32, beta: f32,
) -> (Vec<f32>, Vec<f32>) {
    rk4_step(r, s, dt, gamma, omega, alpha, beta)
}

/// Public wrapper for linear (needed by backward.rs).
#[inline]
pub fn linear_fn(w: &[Vec<f32>], b: &[f32], x: &[f32]) -> Vec<f32> {
    linear(w, b, x)
}

/// Linear without bias: y[i] = sum_j w[i][j] * x[j]
#[inline]
fn linear_no_bias(w: &[Vec<f32>], x: &[f32]) -> Vec<f32> {
    w.iter()
        .map(|row| row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum::<f32>())
        .collect()
}
