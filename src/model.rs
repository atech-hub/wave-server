//! Wave-engine model — CPU reference forward pass for inference.
//!
//! Architecture: parallel blocks (GPT-J formulation)
//!   x = x + attn(LN(x)) + FFN(LN(x))
//!
//! Attention: harmonic coherence scoring — cos(n * Δθ)
//! FFN: dual-maestro Kerr-ODE — maestro_in → ODE → maestro_out → out_proj
//! Embeddings: frozen harmonic table — cos(n*θ) / sin(n*θ)

use std::f32::consts::PI;

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

#[derive(Clone)]
pub struct KerrDualMaestroWeights {
    pub kerr: KerrWeights,
    pub maestro_in: MaestroWeights,
    pub maestro_out: MaestroWeights,
    pub out_proj: LinearWeights,
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

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

// ─── Embedding ──────────────────────────────────────────────

/// Build frozen harmonic embedding table.
pub fn build_harmonic_table(vocab_size: usize, n_bands: usize) -> Vec<Vec<f32>> {
    let n_embd = n_bands * 2;
    (0..vocab_size).map(|tok| {
        let theta = tok as f32 * 2.0 * PI / vocab_size as f32;
        let mut emb = vec![0.0f32; n_embd];
        for n in 0..n_bands {
            let phase = (n + 1) as f32 * theta;
            emb[n * 2] = phase.cos();
            emb[n * 2 + 1] = phase.sin();
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

fn kerr_ode_forward(weights: &KerrWeights, x: &[f32]) -> Vec<f32> {
    let n_bands = weights.gamma_raw.len();
    let n_embd = n_bands * 2;
    let dt = 1.0 / weights.rk4_n_steps as f32;

    let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus(g)).collect();
    let mut r: Vec<f32> = (0..n_bands).map(|k| x[k * 2]).collect();
    let mut s: Vec<f32> = (0..n_bands).map(|k| x[k * 2 + 1]).collect();

    for _ in 0..weights.rk4_n_steps {
        let (k1r, k1s) = kerr_derivative(&r, &s, &gamma, &weights.omega, weights.alpha, weights.beta);
        let r2: Vec<f32> = r.iter().zip(&k1r).map(|(&a,&b)| a+0.5*dt*b).collect();
        let s2: Vec<f32> = s.iter().zip(&k1s).map(|(&a,&b)| a+0.5*dt*b).collect();
        let (k2r, k2s) = kerr_derivative(&r2, &s2, &gamma, &weights.omega, weights.alpha, weights.beta);
        let r3: Vec<f32> = r.iter().zip(&k2r).map(|(&a,&b)| a+0.5*dt*b).collect();
        let s3: Vec<f32> = s.iter().zip(&k2s).map(|(&a,&b)| a+0.5*dt*b).collect();
        let (k3r, k3s) = kerr_derivative(&r3, &s3, &gamma, &weights.omega, weights.alpha, weights.beta);
        let r4: Vec<f32> = r.iter().zip(&k3r).map(|(&a,&b)| a+dt*b).collect();
        let s4: Vec<f32> = s.iter().zip(&k3s).map(|(&a,&b)| a+dt*b).collect();
        let (k4r, k4s) = kerr_derivative(&r4, &s4, &gamma, &weights.omega, weights.alpha, weights.beta);
        for i in 0..n_bands {
            r[i] += dt/6.0 * (k1r[i] + 2.0*k2r[i] + 2.0*k3r[i] + k4r[i]);
            s[i] += dt/6.0 * (k1s[i] + 2.0*k2s[i] + 2.0*k3s[i] + k4s[i]);
        }
    }

    let mut out = vec![0.0f32; n_embd];
    for k in 0..n_bands { out[k * 2] = r[k]; out[k * 2 + 1] = s[k]; }
    out
}

fn dual_maestro_ffn_forward(weights: &KerrDualMaestroWeights, x: &[f32]) -> Vec<f32> {
    let n_embd = x.len();

    // Maestro in (pre-ODE regulator)
    let mae_in = maestro_forward(&weights.maestro_in, x);
    let mut precond = vec![0.0f32; n_embd];
    for i in 0..n_embd { precond[i] = x[i] + mae_in[i]; }

    // Kerr ODE
    let kerr_out = kerr_ode_forward(&weights.kerr, &precond);

    // Maestro out (post-ODE regulator)
    let mae_out = maestro_forward(&weights.maestro_out, &kerr_out);
    let mut regulated = vec![0.0f32; n_embd];
    for i in 0..n_embd { regulated[i] = kerr_out[i] + mae_out[i]; }

    // Out projection
    linear(&weights.out_proj.w, &weights.out_proj.b, &regulated)
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
    pub fn forward_with_memory(
        &self,
        tokens: &[usize],
        memory_offsets: Option<&[(&[f32], &[f32])]>,
    ) -> Vec<Vec<f32>> {
        // TODO: wire memory injection into ODE forward
        // For now, delegates to base forward
        let _ = memory_offsets;
        self.forward(tokens)
    }

    /// Extract ODE states for memory accumulation.
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

            // FFN: capture ODE state before out_proj
            let ffn_out: Vec<Vec<f32>> = normed.iter()
                .map(|x| dual_maestro_ffn_forward(&block.ffn, x))
                .collect();

            // Average ODE states across positions for memory
            let mut avg_r = vec![0.0f32; n_bands];
            let mut avg_s = vec![0.0f32; n_bands];
            for pos in 0..t {
                // Use precond as proxy for ODE input state
                let mae_in = maestro_forward(&block.ffn.maestro_in, &normed[pos]);
                let precond: Vec<f32> = (0..n_embd).map(|i| normed[pos][i] + mae_in[i]).collect();
                for k in 0..n_bands {
                    avg_r[k] += precond[k * 2];
                    avg_s[k] += precond[k * 2 + 1];
                }
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
}
