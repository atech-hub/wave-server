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

fn print_help() {
    println!("wave-server v0.1.0 — OpenAI-compatible API for wave-engine models");
    println!();
    println!("USAGE:");
    println!("    wave-server <CHECKPOINT> [DATA] [OPTIONS]");
    println!();
    println!("ARGUMENTS:");
    println!("    CHECKPOINT      Path to a .bin checkpoint file trained by wave-engine.");
    println!("                    v2 checkpoints are self-describing (architecture stored");
    println!("                    in the header). v1 checkpoints need --n-bands etc.");
    println!();
    println!("    DATA            Path to the training data file (for vocabulary).");
    println!("                    Required for char/word modes — the server rebuilds the");
    println!("                    token mapping from this file. Not needed with --bpe.");
    println!();
    println!("SERVER:");
    println!("    --port N        Listen port                          [default: 8080]");
    println!("    --host ADDR     Bind address. Use 0.0.0.0 to listen  [default: 127.0.0.1]");
    println!("                    on all interfaces (LAN access).");
    println!("    --model-name S  Model name returned in API responses  [default: wave-engine]");
    println!("    --api-key KEY   Require Bearer token authentication. Clients must send");
    println!("                    'Authorization: Bearer <KEY>' header. The /health endpoint");
    println!("                    stays open without auth.              [default: none]");
    println!("    --memory FILE   Load/save wave memory state (.kwmf). Memory offsets inject");
    println!("                    into Kerr-ODE initial conditions. Accumulates experience");
    println!("                    across conversations. Auto-saves after each request.");
    println!("                    Creates fresh file if it doesn't exist. [default: none]");
    println!("    --gpu           Enable GPU acceleration for matmul operations.");
    println!("                    Requires compilation with --features gpu.");
    println!("                    Uses wgpu (Vulkan/Metal/DX12).          [default: off]");
    println!();
    println!("TOKENIZER:");
    println!("    --word          Word-level tokenization. Must match the mode used during");
    println!("                    training. Requires the DATA argument.");
    println!();
    println!("    --bpe FILE      BPE subword tokenization from a HuggingFace tokenizer.json.");
    println!("                    Must match the tokenizer used during training. The DATA");
    println!("                    argument is not needed — vocab comes from the tokenizer file.");
    println!();
    println!("    (default)       Character-level tokenization. Requires the DATA argument.");
    println!();
    println!("ARCHITECTURE (v1 checkpoints only — v2 self-describes):");
    println!("    These flags are only needed for v1 checkpoints that don't store their");
    println!("    architecture in the header. v2 checkpoints (saved by wave-engine v0.2+)");
    println!("    auto-detect all of these.");
    println!();
    println!("    --n-bands N     Harmonic frequency bands              [default: 384]");
    println!("    --n-head N      Attention heads                       [default: 12]");
    println!("    --n-layers N    Transformer blocks                    [default: 24]");
    println!("    --maestro-dim N Maestro bottleneck width               [default: 16]");
    println!("    --block-size N  Max sequence length                    [default: 256]");
    println!("    --rk4-steps N   ODE integration steps per layer        [default: 16]");
    println!();
    println!("ENDPOINTS:");
    println!("    POST /v1/chat/completions    Chat completions (JSON or SSE streaming)");
    println!("    GET  /v1/models              List available models");
    println!("    GET  /health                 Health check (no auth required)");
    println!();
    println!("TELEMETRY:");
    println!("    requests.jsonl  Written automatically for every chat completion request.");
    println!("                    Logs: timestamp, prompt/gen tokens, ms, tokens/sec,");
    println!("                    temperature, top_p, text preview. Survives crashes.");
    println!();
    println!("EXAMPLES:");
    println!("    # Serve a char-level Shakespeare model");
    println!("    wave-server checkpoint_final.bin data/input.txt");
    println!();
    println!("    # Serve with BPE tokenizer (no data file needed)");
    println!("    wave-server checkpoint_final.bin --bpe tokenizer.json");
    println!();
    println!("    # Custom port with API key authentication");
    println!("    wave-server checkpoint_final.bin data/input.txt --port 3000 --api-key mysecret");
    println!();
    println!("    # Listen on all interfaces (LAN access)");
    println!("    wave-server checkpoint_final.bin data/input.txt --host 0.0.0.0");
    println!();
    println!("    # Serve with wave memory (accumulates experience across conversations)");
    println!("    wave-server checkpoint_final.bin data/input.txt --memory memory.kwmf");
    println!();
    println!("    # v1 checkpoint with explicit architecture");
    println!("    wave-server old_checkpoint.bin data/input.txt --n-bands 384 --n-head 12");
    println!();
    println!("CONNECTING CHAT UIs:");
    println!("    Any client that speaks the OpenAI protocol can connect:");
    println!("    - LM Studio: Settings > Local Server > set endpoint to http://127.0.0.1:8080");
    println!("    - Open WebUI: Add connection with base URL http://127.0.0.1:8080");
    println!("    - curl:");
    println!("        curl http://127.0.0.1:8080/v1/chat/completions \\");
    println!("          -H 'Content-Type: application/json' \\");
    println!("          -d '{{\"model\":\"wave-engine\",\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}]}}'");
    println!();
    println!("    Verified with LM Studio 0.4.6. Streaming and non-streaming both supported.");
    println!();
    println!("SOURCE: https://github.com/atech-hub/wave-server");
    println!("ENGINE: https://github.com/atech-hub/wave-engine");
}
