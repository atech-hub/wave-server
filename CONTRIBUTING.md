# Contributing to wave-server

## About this project

This server was built by an IT systems administrator (not a software engineer) collaborating with AI assistants. The architecture and direction come from Marco Da Cunha. Claude Desktop (Opus) handles architecture and documentation. Claude Code handles implementation and testing.

## How to contribute

1. **Fork the repo and create a branch** from `main`.
2. **Make your changes.**
3. **Test all four endpoints:**
   ```bash
   # Build and start
   cargo build --release
   wave-server checkpoint.bin data/input.txt --port 8080

   # In another terminal:
   curl http://localhost:8080/health
   curl http://localhost:8080/v1/models
   curl -X POST http://localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":16}'
   curl -X POST http://localhost:8080/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":16,"stream":true}'
   ```
4. **Submit a PR** with test results and a clear description of what changed.

## What's already built

These features are complete and should not be broken by contributions:

- **OpenAI-compatible API** — chat completions (streaming + non-streaming), models, health
- **KV-cache** — Cached phase angles and value projections for harmonic attention
- **Wave memory** — Persistent experience accumulation via kerr-memory (.kwmf files)
- **BPE tokenizer** — HuggingFace tokenizer.json format
- **WCHK + KCHK checkpoints** — Native wave-engine format + legacy kerr-engine compatibility
- **Bearer token auth** — Optional API key protection
- **LM Studio compatibility** — Verified with 0.4.6

## What needs building

### GPU inference (highest priority)

**Fix gpu_forward.rs** — The GPU forward pass references the old kerr-server architecture (sequential blocks, single maestro, QKV attention). It needs rewriting to match the current wave-engine architecture (parallel blocks, dual maestro, harmonic coherence attention). The `gpu.rs` (GpuAccelerator) is architecture-agnostic — only `gpu_forward.rs` needs updating.

**Batched GPU dispatch** — The current GPU path dispatches one matmul per position in a loop. The wave-engine already has batched dispatch (`linear_batch`). The server should use the same pattern — all positions in one dispatch instead of one-at-a-time.

### Other targets

| Target | Impact | Difficulty |
|--------|--------|------------|
| Vocab embedded in checkpoint | Single-file serving, no data file needed | Small |
| Streaming memory accumulation | Currently only non-streaming updates memory | Small |
| Concurrent request handling | Multi-user serving via batched dispatch | Medium |
| Model hot-reload | Swap models without server restart | Medium |

## Architecture notes

The forward pass uses the **GPT-J parallel block formulation**:
```
x = x + attn(LN(x)) + FFN(LN(x))
```

This matches the training engine exactly. If you modify the forward pass, the parallel vs sequential distinction matters — the model was trained with parallel blocks.

The attention is **frozen** — weights don't change during training or inference. The FFN uses **dual-maestro Kerr-ODE** with maestro_in and maestro_out (both at dim=16). The KV-cache stores phase angles (scalars) and value projections per head — simpler than standard transformer KV-cache because phases are scalars and frozen weights mean cached values never go stale.

## Code of conduct

Be honest about results. If a change makes streaming faster but breaks non-streaming, say so. If GPU inference works on NVIDIA but crashes on AMD, document it. Every honest finding helps.
