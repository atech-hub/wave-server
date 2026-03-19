//! GPU-accelerated forward pass for wave-engine models.
//!
//! Routes large matmuls (out_proj, lm_head) through GpuAccelerator.
//! Harmonic coherence attention scoring stays CPU (phase-based, frozen).
//! Kerr-ODE stays CPU (sequential RK4 integration).
//! Layer norm stays CPU (O(n), not compute-bound).

use crate::gpu::GpuAccelerator;
use crate::model::*;

/// GPU-accelerated forward pass. Matches model.rs CPU forward exactly
/// but routes 768×768 matmuls through GPU.
pub fn forward_gpu(
    model: &ModelWeights,
    tokens: &[usize],
    gpu: &mut GpuAccelerator,
) -> Vec<Vec<f32>> {
    let t = tokens.len();
    let n_embd = model.config.n_embd();
    let n_bands = model.config.n_bands;
    let n_head = model.config.n_head;
    let head_dim = n_embd / n_head;

    // Embed (CPU — lookup)
    let mut hidden: Vec<Vec<f32>> = tokens.iter().enumerate().map(|(pos, &tok)| {
        let mut h = vec![0.0f32; n_embd];
        let tok_idx = tok.min(model.vocab_size - 1);
        let pos_idx = pos.min(model.config.block_size - 1);
        for i in 0..n_embd { h[i] = model.wte[tok_idx][i] + model.wpe[pos_idx][i]; }
        h
    }).collect();

    // Parallel blocks: x = x + attn(LN(x)) + FFN(LN(x))
    for block in &model.blocks {
        let normed: Vec<Vec<f32>> = hidden.iter()
            .map(|h| layer_norm_cpu(h, &block.ln.weight, &block.ln.bias))
            .collect();

        // ─── Attention (harmonic coherence — scoring CPU, out_proj GPU) ───
        let attn_out = harmonic_attention_gpu(&block.attn, &normed, n_bands, n_head, head_dim, gpu);

        // ─── FFN (dual-maestro Kerr-ODE — ODE CPU, out_proj GPU) ───
        let ffn_out: Vec<Vec<f32>> = normed.iter()
            .map(|x| dual_maestro_ffn_gpu(&block.ffn, x, gpu))
            .collect();

        // Parallel residual
        hidden = (0..t).map(|i| {
            let mut v = vec![0.0f32; n_embd];
            for j in 0..n_embd { v[j] = hidden[i][j] + attn_out[i][j] + ffn_out[i][j]; }
            v
        }).collect();
    }

    // Final LN (CPU) + LM head (GPU — vocab×n_embd, biggest matmul)
    hidden.iter().map(|h| {
        let normed = layer_norm_cpu(h, &model.ln_f.weight, &model.ln_f.bias);
        gpu.linear_no_bias(&model.lm_head, &normed)
    }).collect()
}

/// Harmonic coherence attention with GPU out_proj.
/// Phase scoring is CPU (frozen, O(T²×n_heads) of scalar ops).
/// Out_proj is GPU (768×768 matmul per position).
fn harmonic_attention_gpu(
    weights: &WaveAttnWeights,
    x: &[Vec<f32>],
    n_bands: usize,
    n_head: usize,
    head_dim: usize,
    gpu: &mut GpuAccelerator,
) -> Vec<Vec<f32>> {
    let t = x.len();
    let n_embd = n_bands * 2;
    let mut out = vec![vec![0.0f32; n_embd]; t];

    for head in 0..n_head {
        let harmonic_n = softplus_cpu(weights.heads[head].harmonic_raw);
        let offset = head * head_dim;

        // Phase projection (CPU — 2-element output per position)
        let phases: Vec<f32> = (0..t).map(|pos| {
            project_phase_cpu(&x[pos], &weights.heads[head].phase_proj_w, &weights.heads[head].phase_proj_b)
        }).collect();

        // Value projection (CPU — small per-head matmul)
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

        // Phase-hashed sparse attention (CPU — scalar ops)
        const N_BUCKETS: usize = 8;
        let bucket_width = std::f32::consts::TAU / N_BUCKETS as f32;
        let buckets: Vec<usize> = phases.iter().map(|&p| {
            let n = ((p % std::f32::consts::TAU) + std::f32::consts::TAU) % std::f32::consts::TAU;
            ((n / bucket_width) as usize).min(N_BUCKETS - 1)
        }).collect();

        let mut bucket_positions: Vec<Vec<usize>> = vec![Vec::new(); N_BUCKETS];
        for (pos, &b) in buckets.iter().enumerate() { bucket_positions[b].push(pos); }

        for qi in 0..t {
            let qi_bucket = buckets[qi];
            let mut scores = vec![f32::NEG_INFINITY; t];

            for db in 0..=2 {
                let tb = if db == 0 { (qi_bucket + N_BUCKETS - 1) % N_BUCKETS }
                         else if db == 1 { qi_bucket }
                         else { (qi_bucket + 1) % N_BUCKETS };
                for &ki in &bucket_positions[tb] {
                    if ki > qi { continue; }
                    scores[ki] = (harmonic_n * (phases[qi] - phases[ki])).cos();
                }
            }
            if qi > 0 && scores[qi - 1] == f32::NEG_INFINITY {
                scores[qi - 1] = (harmonic_n * (phases[qi] - phases[qi - 1])).cos();
            }
            if scores[qi] == f32::NEG_INFINITY { scores[qi] = 1.0; }

            let max_s = scores[..=qi].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for ki in 0..=qi {
                if scores[ki] > f32::NEG_INFINITY { scores[ki] = (scores[ki] - max_s).exp(); exp_sum += scores[ki]; }
                else { scores[ki] = 0.0; }
            }
            if exp_sum > 0.0 { for ki in 0..=qi { scores[ki] /= exp_sum; } }

            for d in 0..head_dim {
                let mut sum = 0.0f32;
                for ki in 0..=qi { if scores[ki] > 0.0 { sum += scores[ki] * v_all[ki][d]; } }
                out[qi][offset + d] = sum;
            }
        }
    }

    // Output projection — GPU (768×768 per position)
    out.iter().map(|o| gpu.linear(&weights.out_proj_w, &weights.out_proj_b, o)).collect()
}

/// Dual-maestro Kerr-ODE FFN with GPU out_proj.
fn dual_maestro_ffn_gpu(
    weights: &KerrDualMaestroWeights,
    x: &[f32],
    gpu: &mut GpuAccelerator,
) -> Vec<f32> {
    let n_embd = x.len();

    // Maestro in (GPU for projections — dim=16 is small but consistent routing)
    let sq_in = gpu.linear(&weights.maestro_in.squeeze.w, &weights.maestro_in.squeeze.b, x);
    let act_in: Vec<f32> = sq_in.iter().map(|&v| gelu_cpu(v)).collect();
    let mae_in = gpu.linear(&weights.maestro_in.process_1.w, &weights.maestro_in.process_1.b, &act_in);

    let mut precond = vec![0.0f32; n_embd];
    for i in 0..n_embd { precond[i] = x[i] + mae_in[i]; }

    // Kerr ODE (CPU — sequential RK4)
    let kerr_out = kerr_ode_forward_cpu(&weights.kerr, &precond);

    // Maestro out
    let sq_out = gpu.linear(&weights.maestro_out.squeeze.w, &weights.maestro_out.squeeze.b, &kerr_out);
    let act_out: Vec<f32> = sq_out.iter().map(|&v| gelu_cpu(v)).collect();
    let mae_out = gpu.linear(&weights.maestro_out.process_1.w, &weights.maestro_out.process_1.b, &act_out);

    let mut regulated = vec![0.0f32; n_embd];
    for i in 0..n_embd { regulated[i] = kerr_out[i] + mae_out[i]; }

    // Out projection — GPU (768×768)
    gpu.linear(&weights.out_proj.w, &weights.out_proj.b, &regulated)
}

/// Kerr ODE forward (CPU — sequential RK4 integration).
fn kerr_ode_forward_cpu(weights: &KerrWeights, x: &[f32]) -> Vec<f32> {
    let n_bands = weights.gamma_raw.len();
    let n_embd = n_bands * 2;
    let dt = 1.0 / weights.rk4_n_steps as f32;

    let gamma: Vec<f32> = weights.gamma_raw.iter().map(|&g| softplus_cpu(g)).collect();
    let mut r: Vec<f32> = (0..n_bands).map(|k| x[k * 2]).collect();
    let mut s: Vec<f32> = (0..n_bands).map(|k| x[k * 2 + 1]).collect();

    for _ in 0..weights.rk4_n_steps {
        let (k1r, k1s) = kerr_deriv(&r, &s, &gamma, &weights.omega, weights.alpha, weights.beta);
        let r2: Vec<f32> = r.iter().zip(&k1r).map(|(&a,&b)| a+0.5*dt*b).collect();
        let s2: Vec<f32> = s.iter().zip(&k1s).map(|(&a,&b)| a+0.5*dt*b).collect();
        let (k2r, k2s) = kerr_deriv(&r2, &s2, &gamma, &weights.omega, weights.alpha, weights.beta);
        let r3: Vec<f32> = r.iter().zip(&k2r).map(|(&a,&b)| a+0.5*dt*b).collect();
        let s3: Vec<f32> = s.iter().zip(&k2s).map(|(&a,&b)| a+0.5*dt*b).collect();
        let (k3r, k3s) = kerr_deriv(&r3, &s3, &gamma, &weights.omega, weights.alpha, weights.beta);
        let r4: Vec<f32> = r.iter().zip(&k3r).map(|(&a,&b)| a+dt*b).collect();
        let s4: Vec<f32> = s.iter().zip(&k3s).map(|(&a,&b)| a+dt*b).collect();
        let (k4r, k4s) = kerr_deriv(&r4, &s4, &gamma, &weights.omega, weights.alpha, weights.beta);
        for i in 0..n_bands {
            r[i] += dt/6.0 * (k1r[i] + 2.0*k2r[i] + 2.0*k3r[i] + k4r[i]);
            s[i] += dt/6.0 * (k1s[i] + 2.0*k2s[i] + 2.0*k3s[i] + k4s[i]);
        }
    }

    let mut out = vec![0.0f32; n_embd];
    for k in 0..n_bands { out[k * 2] = r[k]; out[k * 2 + 1] = s[k]; }
    out
}

fn kerr_deriv(r: &[f32], s: &[f32], gamma: &[f32], omega: &[f32], alpha: f32, beta: f32) -> (Vec<f32>, Vec<f32>) {
    let n = r.len();
    let mut dr = vec![0.0f32; n];
    let mut ds = vec![0.0f32; n];
    for k in 0..n {
        let mag_sq = r[k]*r[k] + s[k]*s[k];
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

fn layer_norm_cpu(x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    x.iter().zip(weight.iter().zip(bias.iter()))
        .map(|(&xi, (&wi, &bi))| (xi - mean) * inv_std * wi + bi)
        .collect()
}

fn project_phase_cpu(x: &[f32], proj_w: &[Vec<f32>], proj_b: &[f32]) -> f32 {
    let mut r = proj_b[0];
    let mut s = proj_b[1];
    for j in 0..x.len() { r += proj_w[0][j] * x[j]; s += proj_w[1][j] * x[j]; }
    s.atan2(r)
}

fn softplus_cpu(x: f32) -> f32 { if x > 20.0 { x } else { (1.0 + x.exp()).ln() } }
fn gelu_cpu(x: f32) -> f32 { 0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh()) }
