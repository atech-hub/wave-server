//! GPU-accelerated forward pass using the server's own GpuAccelerator.
//!
//! No dependency on kerr-engine. Uses gpu.rs for matmul dispatch.
//! The Kerr-ODE, layer norm, and attention scores stay on CPU
//! (they're O(n) or memory-bound, not the bottleneck at 768-dim+).

use crate::gpu::GpuAccelerator;
use crate::model::*;

/// GPU-backed forward pass with optional wave memory injection.
pub fn forward_with_memory_gpu(
    model: &ModelWeights,
    tokens: &[usize],
    memory_offsets: Option<&[(&[f32], &[f32])]>,
    gpu: &mut GpuAccelerator,
) -> Vec<Vec<f32>> {
    let t = tokens.len();
    let n_embd = model.config.n_embd();
    assert!(t <= model.config.block_size);

    // Embedding + positional encoding (CPU — lookup, not compute)
    let mut hidden: Vec<Vec<f32>> = Vec::with_capacity(t);
    for (pos, &tok) in tokens.iter().enumerate() {
        let mut h = vec![0.0f32; n_embd];
        for i in 0..n_embd { h[i] = model.wte_phase[tok][i] + model.wpe[pos][i]; }
        hidden.push(h);
    }

    // Process through blocks
    let mut ode_layer = 0usize;
    for block in &model.blocks {
        let mem = match (&block.ffn, memory_offsets) {
            (FfnWeights::KerrMaestro(_), Some(offsets)) if ode_layer < offsets.len() => {
                let m = Some(offsets[ode_layer]);
                ode_layer += 1;
                m
            }
            (FfnWeights::KerrMaestro(_), _) => { ode_layer += 1; None }
            _ => None,
        };
        hidden = forward_block_gpu(model, block, &hidden, mem, gpu);
    }

    // Final layer norm (CPU) + LM head (GPU)
    let mut logits = Vec::with_capacity(t);
    for h in &hidden {
        let normed = layer_norm_cpu(h, &model.ln_f.weight, &model.ln_f.bias);
        let l = gpu.linear_no_bias(&model.lm_head, &normed);
        logits.push(l);
    }

    logits
}

fn forward_block_gpu(
    model: &ModelWeights,
    block: &BlockWeights,
    hidden: &[Vec<f32>],
    memory: Option<(&[f32], &[f32])>,
    gpu: &mut GpuAccelerator,
) -> Vec<Vec<f32>> {
    let t = hidden.len();
    let n_embd = model.config.n_embd();

    // Attention: ln1 (CPU) → Q/K/V projection (GPU) → scores (CPU) → out_proj (GPU)
    let normed_1: Vec<Vec<f32>> = hidden.iter()
        .map(|h| layer_norm_cpu(h, &block.ln_1.weight, &block.ln_1.bias))
        .collect();
    let attn_out = attention_gpu(model, &block.attn, &normed_1, gpu);
    let mut h: Vec<Vec<f32>> = (0..t).map(|i| {
        let mut v = vec![0.0f32; n_embd];
        for j in 0..n_embd { v[j] = hidden[i][j] + attn_out[i][j]; }
        v
    }).collect();

    // FFN: ln2 (CPU) → Kerr-ODE (CPU) + Maestro projections (GPU) → out_proj (GPU)
    let normed_2: Vec<Vec<f32>> = h.iter()
        .map(|x| layer_norm_cpu(x, &block.ln_2.weight, &block.ln_2.bias))
        .collect();
    let ffn_out = match &block.ffn {
        FfnWeights::PerBand(w) => per_band_gpu(w, &normed_2, gpu),
        FfnWeights::KerrMaestro(w) => kerr_maestro_gpu(model, w, &normed_2, memory, gpu),
    };
    for i in 0..t {
        for j in 0..n_embd { h[i][j] += ffn_out[i][j]; }
    }

    h
}

/// Attention with GPU-accelerated Q/K/V and output projections.
fn attention_gpu(
    model: &ModelWeights,
    weights: &AttentionWeights,
    x: &[Vec<f32>],
    gpu: &mut GpuAccelerator,
) -> Vec<Vec<f32>> {
    let t = x.len();
    let n_embd = model.config.n_embd();
    let n_head = weights.n_head;
    let head_dim = n_embd / n_head;

    // Q/K/V projection via GPU
    let mut q_all = vec![vec![0.0f32; n_embd]; t];
    let mut k_all = vec![vec![0.0f32; n_embd]; t];
    let mut v_all = vec![vec![0.0f32; n_embd]; t];

    for pos in 0..t {
        let qkv = gpu.linear(&weights.c_attn.w, &weights.c_attn.b, &x[pos]);
        for i in 0..n_embd {
            q_all[pos][i] = qkv[i];
            k_all[pos][i] = qkv[n_embd + i];
            v_all[pos][i] = qkv[2 * n_embd + i];
        }
    }

    // Attention scores + softmax (CPU — memory-bound, not compute-bound)
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![vec![0.0f32; n_embd]; t];

    for head in 0..n_head {
        let offset = head * head_dim;
        for qi in 0..t {
            let mut att = vec![f32::NEG_INFINITY; t];
            for ki in 0..=qi {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_all[qi][offset + d] * k_all[ki][offset + d];
                }
                att[ki] = dot * scale;
            }

            let max_att = att[..=qi].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for ki in 0..=qi {
                att[ki] = (att[ki] - max_att).exp();
                exp_sum += att[ki];
            }
            for ki in 0..=qi { att[ki] /= exp_sum; }

            for d in 0..head_dim {
                let mut val = 0.0f32;
                for ki in 0..=qi {
                    val += att[ki] * v_all[ki][offset + d];
                }
                out[qi][offset + d] = val;
            }
        }
    }

    // Output projection via GPU
    let mut result = Vec::with_capacity(t);
    for pos in 0..t {
        result.push(gpu.linear(&weights.c_proj.w, &weights.c_proj.b, &out[pos]));
    }
    result
}

/// PerBandLinear with GPU output projection.
fn per_band_gpu(
    weights: &PerBandLinearWeights,
    x: &[Vec<f32>],
    gpu: &mut GpuAccelerator,
) -> Vec<Vec<f32>> {
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
            bands_out[band * 2] = w[0][0] * r_in + w[1][0] * s_in + b[0];
            bands_out[band * 2 + 1] = w[0][1] * r_in + w[1][1] * s_in + b[1];
        }
        result.push(gpu.linear(&weights.out_proj.w, &weights.out_proj.b, &bands_out));
    }
    result
}

/// Kerr-ODE + Maestro with memory injection (ODE on CPU, projections on GPU).
fn kerr_maestro_gpu(
    model: &ModelWeights,
    weights: &KerrMaestroAddWeights,
    x: &[Vec<f32>],
    memory: Option<(&[f32], &[f32])>,
    gpu: &mut GpuAccelerator,
) -> Vec<Vec<f32>> {
    let t = x.len();
    let mut result = Vec::with_capacity(t);

    for pos in 0..t {
        // Kerr path (CPU — ODE integration + memory injection)
        let kerr_out = model.kerr_ode_forward_with_memory(&weights.kerr, &x[pos], memory);

        // Maestro path (GPU for projections)
        let squeezed = gpu.linear(&weights.maestro.squeeze.w, &weights.maestro.squeeze.b, &x[pos]);
        let activated: Vec<f32> = squeezed.iter().map(|&v| gelu(v)).collect();
        let maestro_out = gpu.linear(&weights.maestro.process_1.w, &weights.maestro.process_1.b, &activated);

        // Combine + project (GPU)
        let n_embd = kerr_out.len();
        let mut combined = vec![0.0f32; n_embd];
        for i in 0..n_embd { combined[i] = kerr_out[i] + maestro_out[i]; }

        result.push(gpu.linear(&weights.out_proj.w, &weights.out_proj.b, &combined));
    }
    result
}

/// CPU layer norm (not worth GPU dispatch overhead for single vectors).
fn layer_norm_cpu(x: &[f32], weight: &[f32], bias: &[f32]) -> Vec<f32> {
    let n = x.len();
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    x.iter().zip(weight.iter().zip(bias.iter()))
        .map(|(&xi, (&wi, &bi))| (xi - mean) * inv_std * wi + bi)
        .collect()
}

/// GELU activation (CPU).
fn gelu(x: f32) -> f32 {
    let cdf = 0.5 * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh());
    x * cdf
}
