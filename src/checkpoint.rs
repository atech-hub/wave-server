//! Checkpoint loader — load trained model weights from KCHK binary files.
//!
//! Load-only: no save, no optimizer state, no training state.
//! Supports both v1 (needs fallback config) and v2 (self-describing).
//! Inlines the minimum needed from engine's init.rs and optim.rs.

use crate::model::*;

use std::io::{self, Read};
use std::fs::File;

const MAGIC: [u8; 4] = *b"KCHK";

/// Load model weights from a checkpoint file.
/// v2 checkpoints self-describe. v1 uses default_128 config.
pub fn load(path: &str) -> io::Result<ModelWeights> {
    load_with_fallback(path, ModelConfig::default_128())
}

/// Load with explicit fallback config for v1 checkpoints.
pub fn load_with_config(path: &str, fallback: ModelConfig) -> io::Result<ModelWeights> {
    load_with_fallback(path, fallback)
}

fn load_with_fallback(path: &str, fallback: ModelConfig) -> io::Result<ModelWeights> {
    let mut f = File::open(path)?;

    // Header
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Not a kerr-engine checkpoint"));
    }
    let version = read_u32(&mut f)?;

    let config = match version {
        1 => fallback,
        2 => {
            let n_bands = read_u32(&mut f)? as usize;
            let n_head = read_u32(&mut f)? as usize;
            let n_layers = read_u32(&mut f)? as usize;
            let maestro_dim = read_u32(&mut f)? as usize;
            let block_size = read_u32(&mut f)? as usize;
            let rk4_n_steps = read_u32(&mut f)? as usize;
            ModelConfig { n_bands, n_head, n_layers, maestro_dim, block_size, rk4_n_steps }
        }
        _ => {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("Checkpoint version {version}, expected 1 or 2")));
        }
    };

    // Metadata (read but only keep vocab_size)
    let vocab_size = read_u64(&mut f)? as usize;
    let _iter = read_u64(&mut f)?;
    let _lr = read_f32(&mut f)?;
    let _rng_state = read_u64(&mut f)?;

    // Skip Adam state
    let n_params = count_params_for_vocab(vocab_size, &config);
    let _adam_t = read_u64(&mut f)?;
    let _adam_m = read_f32_vec(&mut f, n_params)?;
    let _adam_v = read_f32_vec(&mut f, n_params)?;

    // Model weights
    let params = read_f32_vec(&mut f, n_params)?;
    let mut model = init_model_structure(vocab_size, config);
    unflatten_params(&mut model, &params);

    if version == 2 {
        println!("  Checkpoint v2: {}x{} ({} bands, {} heads, {} layers, maestro {})",
            config.n_embd(), config.n_embd(), config.n_bands, config.n_head,
            config.n_layers, config.maestro_dim);
    } else {
        println!("  Checkpoint v1: using provided config ({} bands)", config.n_bands);
    }

    Ok(model)
}

// ─── Model structure creation (inlined from engine's init.rs) ──

/// Create a ModelWeights with the right shapes, filled with zeros/defaults.
/// The actual values get overwritten by unflatten_params.
fn init_model_structure(vocab_size: usize, config: ModelConfig) -> ModelWeights {
    config.validate();
    let n_embd = config.n_embd();
    let n_bands = config.n_bands;

    // Block 0: PerBandLinear
    let block0 = BlockWeights {
        ln_1: LayerNormWeights { weight: vec![1.0; n_embd], bias: vec![0.0; n_embd] },
        attn: AttentionWeights {
            c_attn: LinearWeights {
                w: vec![vec![0.0; n_embd]; 3 * n_embd],
                b: vec![0.0; 3 * n_embd],
            },
            c_proj: LinearWeights {
                w: vec![vec![0.0; n_embd]; n_embd],
                b: vec![0.0; n_embd],
            },
            n_head: config.n_head,
        },
        ln_2: LayerNormWeights { weight: vec![1.0; n_embd], bias: vec![0.0; n_embd] },
        ffn: FfnWeights::PerBand(PerBandLinearWeights {
            band_w: vec![[[0.0; 2]; 2]; n_bands],
            band_b: vec![[0.0; 2]; n_bands],
            out_proj: LinearWeights {
                w: vec![vec![0.0; n_embd]; n_embd],
                b: vec![0.0; n_embd],
            },
        }),
    };

    // Blocks 1-(n_layers-1): KerrMaestroAdd
    let mut blocks = vec![block0];
    for _ in 0..(config.n_layers - 1) {
        blocks.push(BlockWeights {
            ln_1: LayerNormWeights { weight: vec![1.0; n_embd], bias: vec![0.0; n_embd] },
            attn: AttentionWeights {
                c_attn: LinearWeights {
                    w: vec![vec![0.0; n_embd]; 3 * n_embd],
                    b: vec![0.0; 3 * n_embd],
                },
                c_proj: LinearWeights {
                    w: vec![vec![0.0; n_embd]; n_embd],
                    b: vec![0.0; n_embd],
                },
                n_head: config.n_head,
            },
            ln_2: LayerNormWeights { weight: vec![1.0; n_embd], bias: vec![0.0; n_embd] },
            ffn: FfnWeights::KerrMaestro(KerrMaestroAddWeights {
                kerr: KerrWeights {
                    gamma_raw: vec![0.0; n_bands],
                    omega: vec![0.0; n_bands],
                    alpha: 0.0,
                    beta: 0.0,
                    rk4_n_steps: config.rk4_n_steps,
                },
                maestro: MaestroWeights {
                    squeeze: LinearWeights {
                        w: vec![vec![0.0; n_embd]; config.maestro_dim],
                        b: vec![0.0; config.maestro_dim],
                    },
                    process_1: LinearWeights {
                        w: vec![vec![0.0; config.maestro_dim]; n_embd],
                        b: vec![0.0; n_embd],
                    },
                },
                out_proj: LinearWeights {
                    w: vec![vec![0.0; n_embd]; n_embd],
                    b: vec![0.0; n_embd],
                },
            }),
        });
    }

    ModelWeights {
        config,
        vocab_size,
        wte_phase: build_harmonic_table_sized(vocab_size, n_embd),
        wpe: build_positional_table(config.block_size, n_embd),
        blocks,
        ln_f: LayerNormWeights { weight: vec![1.0; n_embd], bias: vec![0.0; n_embd] },
        lm_head: vec![vec![0.0; n_embd]; vocab_size],
    }
}

// ─── Parameter counting (inlined from engine's optim.rs) ──────

/// Count total trainable parameters for a given vocab size and config.
pub fn count_params_for_vocab(vocab_size: usize, config: &ModelConfig) -> usize {
    let n_embd = config.n_embd();
    let n_bands = config.n_bands;
    let maestro_dim = config.maestro_dim;
    let mut n = 0;

    // Block 0: PerBandLinear
    n += n_embd * 2; // ln_1
    n += 3 * n_embd * n_embd + 3 * n_embd; // c_attn
    n += n_embd * n_embd + n_embd; // c_proj
    n += n_embd * 2; // ln_2
    n += n_bands * 4 + n_bands * 2; // band_w + band_b
    n += n_embd * n_embd + n_embd; // out_proj

    // Blocks 1-(n_layers-1): KerrMaestro
    for _ in 0..(config.n_layers - 1) {
        n += n_embd * 2; // ln_1
        n += 3 * n_embd * n_embd + 3 * n_embd; // c_attn
        n += n_embd * n_embd + n_embd; // c_proj
        n += n_embd * 2; // ln_2
        n += n_bands + n_bands + 1 + 1; // gamma_raw, omega, alpha, beta
        n += maestro_dim * n_embd + maestro_dim; // squeeze
        n += n_embd * maestro_dim + n_embd; // process
        n += n_embd * n_embd + n_embd; // out_proj
    }

    n += n_embd * 2; // ln_f
    n += vocab_size * n_embd; // lm_head
    n
}

/// Count params for an existing model.
pub fn count_params(model: &ModelWeights) -> usize {
    count_params_for_vocab(model.vocab_size, &model.config)
}

// ─── Parameter unflattening (inlined from engine's optim.rs) ──

fn unflatten_params(model: &mut ModelWeights, params: &[f32]) {
    let n_embd = model.config.n_embd();
    let mut idx = 0;

    for block in &mut model.blocks {
        block.ln_1.weight.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
        block.ln_1.bias.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
        for row in &mut block.attn.c_attn.w {
            row.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
        }
        let attn_b_len = block.attn.c_attn.b.len();
        block.attn.c_attn.b.copy_from_slice(&params[idx..idx + attn_b_len]); idx += attn_b_len;
        for row in &mut block.attn.c_proj.w {
            row.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
        }
        block.attn.c_proj.b.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
        block.ln_2.weight.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
        block.ln_2.bias.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;

        match &mut block.ffn {
            FfnWeights::PerBand(w) => {
                for band in &mut w.band_w {
                    band[0].copy_from_slice(&params[idx..idx + 2]); idx += 2;
                    band[1].copy_from_slice(&params[idx..idx + 2]); idx += 2;
                }
                for band in &mut w.band_b {
                    band.copy_from_slice(&params[idx..idx + 2]); idx += 2;
                }
                for row in &mut w.out_proj.w {
                    row.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
                }
                w.out_proj.b.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
            }
            FfnWeights::KerrMaestro(w) => {
                let n_bands = w.kerr.gamma_raw.len();
                let maestro_dim = w.maestro.squeeze.b.len();
                w.kerr.gamma_raw.copy_from_slice(&params[idx..idx + n_bands]); idx += n_bands;
                w.kerr.omega.copy_from_slice(&params[idx..idx + n_bands]); idx += n_bands;
                w.kerr.alpha = params[idx]; idx += 1;
                w.kerr.beta = params[idx]; idx += 1;
                for row in &mut w.maestro.squeeze.w {
                    row.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
                }
                w.maestro.squeeze.b.copy_from_slice(&params[idx..idx + maestro_dim]); idx += maestro_dim;
                for row in &mut w.maestro.process_1.w {
                    row.copy_from_slice(&params[idx..idx + maestro_dim]); idx += maestro_dim;
                }
                w.maestro.process_1.b.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
                for row in &mut w.out_proj.w {
                    row.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
                }
                w.out_proj.b.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
            }
        }
    }

    model.ln_f.weight.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
    model.ln_f.bias.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
    for row in &mut model.lm_head {
        row.copy_from_slice(&params[idx..idx + n_embd]); idx += n_embd;
    }

    assert_eq!(idx, params.len(), "Param count mismatch in unflatten");
}

// ─── Binary helpers ───────────────────────────────────────────

fn read_u32(f: &mut File) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(f: &mut File) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f32(f: &mut File) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    f.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f32_vec(f: &mut File, expected_len: usize) -> io::Result<Vec<f32>> {
    let len = read_u64(f)? as usize;
    if len != expected_len {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("Expected {expected_len} floats, got {len}")));
    }
    let mut data = vec![0.0f32; len];
    for v in &mut data {
        *v = read_f32(f)?;
    }
    Ok(data)
}
