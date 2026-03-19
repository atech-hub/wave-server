# wave-server

OpenAI-compatible API server for [wave-engine](https://github.com/atech-hub/wave-engine) models. Load a trained checkpoint, serve it via HTTP, optionally accumulate wave memory across conversations. Any chat UI that speaks the OpenAI protocol connects without modification.

Part of the [Wave Coherence as a Computational Primitive](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive) research project.

## Quick Start

```bash
# Build
cargo build --release

# Serve a model (WCHK checkpoint — self-describing, no config flags needed)
wave-server checkpoint.bin data/input.txt --port 8080

# Serve with BPE tokenizer (no data file needed)
wave-server checkpoint.bin --bpe tokenizer.json --port 8080

# Serve with wave memory (accumulates experience across conversations)
wave-server checkpoint.bin data/input.txt --memory memory.kwmf --port 8080

# Serve with API key authentication
wave-server checkpoint.bin data/input.txt --port 8080 --api-key sk-your-secret-key

# Test
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

## Usage

```
wave-server <checkpoint> [data] [options]

Arguments:
  <checkpoint>      Path to checkpoint file (.bin) trained by wave-engine
  [data]            Path to training data (for vocabulary — optional with --bpe)

Tokenizer (one required):
  [data]            Character/word vocab extracted from training data
  --bpe FILE        BPE tokenizer from HuggingFace tokenizer.json
  --word            Word-level tokenizer (requires data file)

Server:
  --port N          Listen port (default: 8080)
  --host ADDR       Bind address (default: 127.0.0.1, use 0.0.0.0 for LAN)
  --model-name S    Model name in API responses (default: wave-engine)
  --api-key KEY     Require bearer token auth on /v1/* endpoints
  --gpu             Enable GPU acceleration (requires --features gpu build)

Wave Memory:
  --memory FILE     Load/create a .kwmf wave memory file. Injects into
                    ODE initial conditions. Accumulates across conversations.

Architecture (WCHK v1 checkpoints only — v2 self-describes):
  --n-bands N       Harmonic frequency bands (default: 384)
  --n-head N        Attention heads (default: 12)
  --n-layers N      Parallel blocks (default: 24)
  --maestro-dim N   Maestro bottleneck width (default: 16)
  --block-size N    Max sequence length (default: 256)
  --rk4-steps N     ODE integration steps (default: 16)
```

## Endpoints

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/v1/chat/completions` | POST | Yes | OpenAI-compatible chat completion |
| `/v1/models` | GET | Yes | List available models |
| `/health` | GET | No | Server health check with model info |

**POST /v1/chat/completions** — Accepts `messages` (required), `temperature`, `top_p`, `max_tokens`, `stream`, `top_k`, `repetition_penalty`. Non-streaming returns JSON. Streaming (`"stream": true`) returns Server-Sent Events ending with `data: [DONE]\n\n`.

**Auth** — When `--api-key` is set, all `/v1/*` endpoints require `Authorization: Bearer <key>` header. `/health` is always open for load balancers and monitoring.

## Checkpoint Formats

**WCHK (recommended)** — Wave-engine native format. Self-describing header contains the full model config. No architecture flags needed.

```bash
wave-server checkpoint.bin data/input.txt --port 8080
```

**KCHK (legacy)** — Kerr-engine format. Loaded with best-effort compatibility. Architecture flags may be needed if the model doesn't match defaults.

```bash
wave-server legacy_checkpoint.bin data/input.txt --n-bands 64 --n-head 4 --n-layers 4
```

## Connecting Chat UIs

Any chat application that supports OpenAI-compatible endpoints can connect.

**LM Studio** (verified with 0.4.6) — In the chat panel, set "Override Base URL" to `http://127.0.0.1:8080/v1`. Select any model from the dropdown — the server uses its loaded model regardless. Put your API key in the "OpenAI API Key" field (or any string if auth is disabled).

**Open WebUI** — Settings → Connections → add `http://127.0.0.1:8080/v1` as an OpenAI endpoint.

**SillyTavern / continue.dev / any OpenAI client** — Point the API base URL to `http://127.0.0.1:8080/v1`. Set the API key if auth is enabled.

**curl** —
```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-key" \
  -d '{
    "model": "wave-engine",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
    "temperature": 0.8,
    "stream": true
  }'
```

## Wave Memory

The `--memory` flag enables persistent experience accumulation across conversations. The model weights never change — a separate file (typically 1-2KB) shifts the Kerr-ODE's starting position on the unit circle.

```bash
# First conversation — creates fresh memory file
wave-server checkpoint.bin data/input.txt --memory memory.kwmf

# Subsequent conversations — loads and accumulates
wave-server checkpoint.bin data/input.txt --memory memory.kwmf

# Inspect what accumulated
kerr-memory census memory.kwmf
```

During inference, memory offsets add to the ODE initial conditions: `Z_k = input_k + α · memory_k`. After each conversation, the ODE final states feed an exponential moving average that merges into the persistent file. The Kerr dynamics amplify resonant memories and damp unreinforced ones.

The memory system is model-agnostic — it stores per-band oscillator states and works with any model that uses the Kerr-ODE. See [kerr-memory](https://github.com/atech-hub/kerr-memory) for the library and full investigation results.

## Architecture

The server implements the wave-engine forward pass for inference. The architecture matches the training engine exactly — parallel blocks with harmonic coherence attention and dual-maestro Kerr-ODE FFN.

**Block structure (GPT-J parallel formulation):**
```
x = x + attention(LN(x)) + FFN(LN(x))
```

**Attention** — Harmonic coherence scoring: `cos(n × Δφ)` where `n` is a learned harmonic number and `Δφ` is the phase difference between positions. Frozen during training, frozen during inference.

**FFN** — Dual-maestro Kerr-ODE: `input → maestro_in (768→16→768) → ODE (RK4) → maestro_out (768→16→768) → out_proj (768→768)`.

~1,500 lines across 12 modules. Self-contained — the forward pass is built in, no wave-engine dependency at runtime.

```
src/
├── main.rs          CLI, startup, config
├── server.rs        Axum router, auth middleware, graceful shutdown
├── handlers.rs      Request handlers, SSE streaming, memory accumulation
├── api_types.rs     OpenAI protocol types (serde structs)
├── model.rs         Forward pass, weight structs, Kerr-ODE with memory injection
├── checkpoint.rs    Checkpoint loader (WCHK and legacy KCHK)
├── inference.rs     Token generation with temperature/top-k/top-p sampling
├── prompt.rs        Vocabulary, text encode/decode, chat message formatting
├── data.rs          Character and word tokenizers
├── bpe.rs           BPE tokenizer (HuggingFace tokenizer.json)
├── rng.rs           Deterministic PRNG for sampling
├── gpu.rs           GPU matvec accelerator (optional, wgpu)
└── gpu_forward.rs   GPU-accelerated forward pass (optional)
```

## Known Limitations

**No KV-cache** — The server re-runs the full forward pass for every new token generated. At 24 layers / 768-dim, this means ~1-2 seconds per token — a 100-token response takes minutes. This is the #1 performance bottleneck. At 4 layers / 128-dim, inference is instant (~1ms per token).

**CPU-only by default** — The GPU feature (`--features gpu`) accelerates matmul operations via wgpu but the ODE and attention scoring remain on CPU. For small models this is fine. For large models, GPU inference with KV-cache is needed.

**No concurrent requests** — The server handles one request at a time. Sequential inference.

## Dependencies

- [kerr-memory](https://github.com/atech-hub/kerr-memory) — Wave memory state management (for `--memory` flag)
- [axum](https://github.com/tokio-rs/axum) 0.8 — HTTP framework
- [tokio](https://tokio.rs) — Async runtime
- serde, serde_json, uuid, tokio-stream
- wgpu, bytemuck, pollster (optional, for `--features gpu`)

No wave-engine dependency at runtime. The forward pass is built into the server.

## Requirements

- Rust (edition 2024)
- kerr-memory repo at `../kerr-memory` (path dependency for wave memory support)

```bash
# Default build (CPU inference)
cargo build --release

# With GPU acceleration
cargo build --release --features gpu
```

## Contributing

The maintainer ([Marco Da Cunha](https://github.com/atech-hub)) is an IT systems administrator, not a programmer. The server was built through collaboration with AI — Claude Desktop for architecture, Claude Code for implementation. This is stated openly.

**Contribution process:**
- Fork the repo, create a branch, submit a PR.
- Every PR must demonstrate that the four endpoints still work: health, models, non-streaming chat, and SSE streaming.
- The maintainer merges based on testing and description, not code review. Be clear about what you changed and why.

**Known targets for contributors (priority order):**

| Target | Impact | Difficulty |
|--------|--------|------------|
| **KV-cache** | 100x faster generation at 768-dim+ | Medium-Large |
| **GPU inference pipeline** | Full forward pass on GPU | Medium |
| Vocab embedded in checkpoint | Eliminate data file requirement | Small |
| Streaming memory accumulation | Currently only non-streaming updates memory | Small |
| Concurrent request handling | Multi-user support | Medium |
| Model hot-reload | Swap models without restart | Medium |

## Related

- [wave-engine](https://github.com/atech-hub/wave-engine) — Training engine that produces the checkpoints this server serves (public, Apache 2.0)
- [kerr-memory](https://github.com/atech-hub/kerr-memory) — Wave memory library used by the `--memory` flag (public, Apache 2.0)
- [Wave Coherence as a Computational Primitive](https://github.com/atech-hub/Wave-Coherence-as-a-Computational-Primitive) — The parent research project (public, MIT)

## License

Apache 2.0. See [LICENSE](LICENSE).

## Credits

- **Marco Da Cunha** — Architecture, direction, pattern recognition
- **Claude Desktop (Opus)** — Architecture design, documentation
- **Claude Code** — Implementation, testing
