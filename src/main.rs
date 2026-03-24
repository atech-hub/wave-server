//! wave-server — OpenAI-compatible API for wave-engine models.
//!
//! Usage:
//!   wave-server <checkpoint> [data] [options]
//!
//! v2 checkpoints self-describe — just point at the file.
//! v1 checkpoints need architecture flags to match the training config.
//! BPE mode (--bpe tokenizer.json) does not require a data file.

mod api_types;
mod bpe;
mod checkpoint;
mod data;
mod handlers;
mod help;
mod inference;
mod model;
mod prompt;
mod rng;
mod server;

#[cfg(feature = "gpu")]
mod gpu;
#[cfg(feature = "gpu")]
mod gpu_forward;

use std::sync::Arc;

use crate::bpe::BpeTokenizer;
use crate::data::Dataset;
use crate::model::ModelConfig;
use crate::prompt::Vocab;
use crate::server::{AppState, ServerConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 || args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        std::process::exit(if args.len() < 2 { 1 } else { 0 });
    }

    let checkpoint_path = &args[1];

    // Parse optional flags — need to scan for --bpe before determining if data arg is required
    let mut port: u16 = 8080;
    let mut host = "127.0.0.1".to_string();
    let mut model_name = "wave-engine".to_string();
    let mut api_key: Option<String> = None;
    let mut word_level = false;
    let mut bpe_path: Option<String> = None;
    let mut memory_path: Option<String> = None;
    let mut use_gpu = false;
    let mut config = ModelConfig::default_768();
    let mut has_arch_flags = false;

    // First pass: find --bpe and --memory before positional arg parsing
    for (i, arg) in args.iter().enumerate() {
        if arg == "--bpe" {
            bpe_path = args.get(i + 1).cloned();
        }
        if arg == "--memory" {
            memory_path = args.get(i + 1).cloned();
        }
    }

    // Determine where flags start and if data path is provided
    let (data_path, flags_start) = if bpe_path.is_some() {
        // With --bpe, data arg is optional. Check if arg[2] looks like a flag or a file.
        if args.len() > 2 && !args[2].starts_with("--") {
            (Some(args[2].clone()), 3)
        } else {
            (None, 2)
        }
    } else {
        // Without --bpe, data arg is required
        if args.len() < 3 {
            eprintln!("ERROR: <data> argument required (or use --bpe for BPE tokenizer)");
            std::process::exit(1);
        }
        (Some(args[2].clone()), 3)
    };

    let mut i = flags_start;
    while i < args.len() {
        match args[i].as_str() {
            "--port" => { i += 1; port = args[i].parse().expect("invalid port"); }
            "--host" => { i += 1; host = args[i].clone(); }
            "--model-name" => { i += 1; model_name = args[i].clone(); }
            "--api-key" => { i += 1; api_key = Some(args[i].clone()); }
            "--word" => { word_level = true; }
            "--gpu" => { use_gpu = true; }
            "--bpe" => { i += 1; } // already parsed in first pass
            "--memory" => { i += 1; } // already parsed in first pass
            "--n-bands" => { i += 1; config.n_bands = args[i].parse().expect("invalid n-bands"); has_arch_flags = true; }
            "--n-head" => { i += 1; config.n_head = args[i].parse().expect("invalid n-head"); has_arch_flags = true; }
            "--n-layers" => { i += 1; config.n_layers = args[i].parse().expect("invalid n-layers"); has_arch_flags = true; }
            "--maestro-dim" => { i += 1; config.maestro_dim = args[i].parse().expect("invalid maestro-dim"); has_arch_flags = true; }
            "--block-size" => { i += 1; config.block_size = args[i].parse().expect("invalid block-size"); has_arch_flags = true; }
            "--rk4-steps" => { i += 1; config.rk4_n_steps = args[i].parse().expect("invalid rk4-steps"); has_arch_flags = true; }
            other => { eprintln!("Unknown flag: {other}"); std::process::exit(1); }
        }
        i += 1;
    }

    config.validate();

    // Load model from checkpoint
    println!("Loading checkpoint: {checkpoint_path}");
    let config_arg = if has_arch_flags { Some(config) } else { None };
    let (model, _iter, _lr) = checkpoint::load_checkpoint(checkpoint_path, config_arg);
    println!("  Model: {} layers, {} embd ({} bands), {} vocab",
        model.config.n_layers, model.config.n_embd(), model.config.n_bands,
        model.vocab_size);

    // Load vocabulary
    let vocab = if let Some(ref bpe_file) = bpe_path {
        println!("Loading BPE tokenizer from: {bpe_file}");
        let bpe = BpeTokenizer::from_file(bpe_file);
        let v = Vocab::from_bpe(bpe);
        println!("  Vocab: {} tokens, mode=bpe", v.vocab_size);
        v
    } else {
        let dp = data_path.as_ref().unwrap();
        println!("Loading vocabulary from: {dp}");
        let dataset = if word_level {
            Dataset::from_file_words(dp, 0.9, 3)
        } else {
            Dataset::from_file(dp)
        };
        let v = Vocab::from_dataset(&dataset);
        println!("  Vocab: {} tokens, mode={}",
            v.vocab_size, if word_level { "word" } else { "char" });
        v
    };

    // Verify vocab sizes match
    if vocab.vocab_size != model.vocab_size {
        eprintln!("WARNING: vocab size mismatch — model={}, tokenizer={}",
            model.vocab_size, vocab.vocab_size);
        eprintln!("  Model was trained with different tokenizer. Results may be incorrect.");
    }

    // Load or create wave memory
    let memory = if let Some(ref path) = memory_path {
        if std::path::Path::new(path).exists() {
            println!("Loading wave memory from: {path}");
            let mem = kerr_memory::file::load(path).expect("Failed to load memory file");
            println!("  Memory: {} layers, {} bands, {} conversations",
                mem.n_layers(), mem.n_bands, mem.n_convos);
            Some(mem)
        } else {
            println!("Creating new wave memory: {path}");
            let n_ode_layers = model.config.n_layers - 1; // Block 0 is PerBandLinear
            let mem = kerr_memory::memory::WaveMemory::zeros(n_ode_layers, model.config.n_bands);
            println!("  Memory: {} layers, {} bands (fresh)", n_ode_layers, model.config.n_bands);
            Some(mem)
        }
    } else {
        None
    };

    // GPU inference setup
    #[cfg(feature = "gpu")]
    let gpu_accel = if use_gpu {
        println!("Initialising GPU...");
        let max_dim = model.config.n_embd().max(model.vocab_size);
        Some(gpu::GpuAccelerator::new(max_dim))
    } else { None };

    #[cfg(not(feature = "gpu"))]
    if use_gpu {
        eprintln!("WARNING: --gpu requires: cargo build --release --features gpu");
        eprintln!("  Falling back to CPU inference.");
    }

    let app_state = Arc::new(AppState {
        model: Arc::new(model),
        vocab: Arc::new(vocab),
        config: ServerConfig { host, port, model_name, api_key },
        memory: std::sync::Mutex::new(memory),
        memory_path: memory_path.clone(),
        #[cfg(feature = "gpu")]
        gpu: std::sync::Mutex::new(gpu_accel),
    });

    // Start tokio runtime and run server
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(server::run(app_state));
}

fn print_help() { help::print_help(); }
