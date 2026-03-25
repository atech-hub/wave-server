//! Wave-engine checkpoint loader.
//!
//! Supports two formats:
//! 1. WCHK v1: wave-engine native binary (header + raw params)
//! 2. Legacy KCHK: kerr-engine format (backward compatible, best-effort)
//!
//! WCHK format:
//!   "WCHK" [4 bytes] magic
//!   version [u32] = 1
//!   n_bands, n_head, n_layers, maestro_dim, block_size, rk4_n_steps [6 × u32]
//!   vocab_size [u64], iter [u64], lr [f32], rng_state [u64]
//!   adam_t [u64], adam_m [n f32s], adam_v [n f32s] (skipped)
//!   params [n_trainable f32s]

use crate::model::*;
use std::io::{Read, Seek, SeekFrom};

/// Load a wave-engine checkpoint from a binary file.
pub fn load_checkpoint(
    path: &str,
    config_override: Option<ModelConfig>,
) -> (ModelWeights, u64, f32) {
    let mut file = std::fs::File::open(path)
        .unwrap_or_else(|e| panic!("Failed to open checkpoint {path}: {e}"));

    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).expect("Failed to read magic");

    match &magic {
        b"WCHK" => load_wchk(&mut file, config_override),
        b"KCHK" => {
            eprintln!("  Warning: KCHK (kerr-engine) format — architecture may differ");
            file.seek(SeekFrom::Start(0)).unwrap();
            load_kchk(&mut file, config_override)
        }
        _ => panic!("Unknown checkpoint format: {:?}", std::str::from_utf8(&magic).unwrap_or("???"))
    }
}

fn read_u32(f: &mut std::fs::File) -> u32 {
    let mut buf = [0u8; 4]; f.read_exact(&mut buf).unwrap(); u32::from_le_bytes(buf)
}
fn read_u64(f: &mut std::fs::File) -> u64 {
    let mut buf = [0u8; 8]; f.read_exact(&mut buf).unwrap(); u64::from_le_bytes(buf)
}
fn read_f32_single(f: &mut std::fs::File) -> f32 {
    let mut buf = [0u8; 4]; f.read_exact(&mut buf).unwrap(); f32::from_le_bytes(buf)
}
fn read_f32_vec(f: &mut std::fs::File, n: usize) -> Vec<f32> {
    let mut buf = vec![0u8; n * 4];
    f.read_exact(&mut buf).unwrap();
    buf.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn load_wchk(file: &mut std::fs::File, config_override: Option<ModelConfig>) -> (ModelWeights, u64, f32) {
    let version = read_u32(file);
    assert!(version == 1 || version == 2, "Unknown WCHK version {version}");

    let (config, out_proj_groups) = if let Some(c) = config_override {
        // Skip config fields: 6 for v1, 7 for v2
        let n_fields = if version >= 2 { 7 } else { 6 };
        for _ in 0..n_fields { read_u32(file); }
        (c, 6) // default groups for override
    } else {
        let c = ModelConfig {
            n_bands: read_u32(file) as usize,
            n_head: read_u32(file) as usize,
            n_layers: read_u32(file) as usize,
            maestro_dim: read_u32(file) as usize,
            block_size: read_u32(file) as usize,
            rk4_n_steps: read_u32(file) as usize,
        };
        let out_proj_groups = if version >= 2 { read_u32(file) as usize } else { 6 };
        (c, out_proj_groups)
    };
    config.validate();

    let vocab_size = read_u64(file) as usize;
    let iter = read_u64(file);
    let lr = read_f32_single(file);
    let _rng_state = read_u64(file);

    let n_trainable = count_trainable_params(&config, vocab_size, out_proj_groups);

    // Skip optimizer state (adam_t + m + v)
    let _adam_t = read_u64(file);
    file.seek(SeekFrom::Current((n_trainable * 2 * 4) as i64)).unwrap();

    let params = read_f32_vec(file, n_trainable);
    let model = unflatten_to_model(&config, vocab_size, out_proj_groups, &params);

    println!("  WCHK checkpoint: {} params, iter {iter}, lr {lr:.6}", params.len());
    println!("  Config: {}bands × {}head × {}layers = {}dim",
        config.n_bands, config.n_head, config.n_layers, config.n_embd());

    (model, iter, lr)
}

fn load_kchk(file: &mut std::fs::File, config_override: Option<ModelConfig>) -> (ModelWeights, u64, f32) {
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).unwrap();
    let version = read_u32(file);

    let config = if version >= 2 {
        let c = ModelConfig {
            n_bands: read_u32(file) as usize,
            n_head: read_u32(file) as usize,
            n_layers: read_u32(file) as usize,
            maestro_dim: read_u32(file) as usize,
            block_size: read_u32(file) as usize,
            rk4_n_steps: read_u32(file) as usize,
        };
        config_override.unwrap_or(c)
    } else {
        config_override.unwrap_or_else(ModelConfig::default_768)
    };

    let vocab_size = read_u64(file) as usize;
    let iter = read_u64(file);
    let lr = read_f32_single(file);

    eprintln!("  KCHK v{version}: vocab={vocab_size}, iter={iter} — using empty weights (architecture mismatch)");
    let model = create_empty_model(&config, vocab_size, 6); // KCHK default
    (model, iter, lr)
}

/// Count trainable parameters (matches wave-engine's flatten_params layout).
fn count_trainable_params(config: &ModelConfig, vocab_size: usize, out_proj_groups: usize) -> usize {
    let n_embd = config.n_embd();
    let md = config.maestro_dim;
    let gs = n_embd / out_proj_groups.max(1);

    let per_block =
        n_embd * 2 +              // ln
        n_embd * 2 +              // ln_ffn
        md * n_embd + md +        // mae_in squeeze
        n_embd * md + n_embd +    // mae_in process
        md * n_embd + md +        // mae_out squeeze
        n_embd * md + n_embd +    // mae_out process
        out_proj_groups * (gs * gs + gs); // block-diagonal out_proj

    config.n_layers * per_block + n_embd * 2 + vocab_size * n_embd
}

fn create_empty_model(config: &ModelConfig, vocab_size: usize, out_proj_groups: usize) -> ModelWeights {
    let n_embd = config.n_embd();
    let n_bands = config.n_bands;
    let md = config.maestro_dim;
    let head_dim = n_embd / config.n_head;

    let make_linear = |out: usize, inp: usize| LinearWeights {
        w: vec![vec![0.0f32; inp]; out], b: vec![0.0f32; out],
    };

    let blocks = (0..config.n_layers).map(|_| {
        let heads = (0..config.n_head).map(|i| WaveAttnHeadWeights {
            harmonic_raw: ((i + 1) as f32 * 0.5f32).ln(),
            phase_proj_w: vec![vec![0.0f32; n_embd]; 2],
            phase_proj_b: vec![0.0f32; 2],
            v_proj_w: vec![vec![0.0f32; head_dim]; head_dim],
            v_proj_b: vec![0.0f32; head_dim],
        }).collect();
        WaveBlockWeights {
            ln: LayerNormWeights { weight: vec![1.0; n_embd], bias: vec![0.0; n_embd] },
            ln_ffn: LayerNormWeights { weight: vec![1.0; n_embd], bias: vec![0.0; n_embd] },
            attn: WaveAttnWeights { heads, out_proj_w: vec![vec![0.0; n_embd]; n_embd], out_proj_b: vec![0.0; n_embd] },
            ffn: KerrDualMaestroWeights {
                kerr: KerrWeights {
                    gamma_raw: vec![0.0; n_bands],
                    omega: (0..n_bands).map(|k| (k + 1) as f32 / n_bands as f32).collect(),
                    alpha: 0.1, beta: 0.1, rk4_n_steps: config.rk4_n_steps,
                },
                maestro_in: MaestroWeights { squeeze: make_linear(md, n_embd), process_1: make_linear(n_embd, md) },
                maestro_out: MaestroWeights { squeeze: make_linear(md, n_embd), process_1: make_linear(n_embd, md) },
                out_proj: {
                    let n_groups = out_proj_groups;
                    let group_size = n_embd / n_groups;
                    BlockDiagonalWeights {
                        groups: (0..n_groups).map(|_| make_linear(group_size, group_size)).collect(),
                        n_groups,
                        group_size,
                    }
                },
            },
        }
    }).collect();

    ModelWeights {
        config: *config, vocab_size,
        wte: build_harmonic_table(vocab_size, n_bands),
        wpe: build_positional_table(config.block_size, n_bands),
        blocks,
        ln_f: LayerNormWeights { weight: vec![1.0; n_embd], bias: vec![0.0; n_embd] },
        lm_head: vec![vec![0.0f32; n_embd]; vocab_size],
    }
}

fn unflatten_to_model(config: &ModelConfig, vocab_size: usize, out_proj_groups: usize, params: &[f32]) -> ModelWeights {
    let mut model = create_empty_model(config, vocab_size, out_proj_groups);
    let n_embd = config.n_embd();
    let md = config.maestro_dim;
    let mut idx = 0;

    for block in &mut model.blocks {
        block.ln.weight.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
        block.ln.bias.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
        block.ln_ffn.weight.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
        block.ln_ffn.bias.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
        for row in &mut block.ffn.maestro_in.squeeze.w { row.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd; }
        block.ffn.maestro_in.squeeze.b.copy_from_slice(&params[idx..idx+md]); idx += md;
        for row in &mut block.ffn.maestro_in.process_1.w { row.copy_from_slice(&params[idx..idx+md]); idx += md; }
        block.ffn.maestro_in.process_1.b.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
        for row in &mut block.ffn.maestro_out.squeeze.w { row.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd; }
        block.ffn.maestro_out.squeeze.b.copy_from_slice(&params[idx..idx+md]); idx += md;
        for row in &mut block.ffn.maestro_out.process_1.w { row.copy_from_slice(&params[idx..idx+md]); idx += md; }
        block.ffn.maestro_out.process_1.b.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
        // Block-diagonal out_proj: groups from checkpoint header
        let gs = block.ffn.out_proj.group_size;
        for g in &mut block.ffn.out_proj.groups {
            for row in &mut g.w { row.copy_from_slice(&params[idx..idx+gs]); idx += gs; }
            g.b.copy_from_slice(&params[idx..idx+gs]); idx += gs;
        }
    }

    model.ln_f.weight.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
    model.ln_f.bias.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd;
    for row in &mut model.lm_head { row.copy_from_slice(&params[idx..idx+n_embd]); idx += n_embd; }

    assert_eq!(idx, params.len(), "Param count mismatch: used {idx}, have {}", params.len());
    model
}
