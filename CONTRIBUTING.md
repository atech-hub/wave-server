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

## What needs building

### KV-cache (critical for larger models)

Without KV-cache, the server re-runs the full forward pass on the entire context for every new token. At 24 layers / 768-dim, each token takes ~1-2 seconds — chat is unusable.

With KV-cache, only the new token position runs through the model. Cached K/V from previous positions are reused. This is especially effective with the frozen harmonic attention — cached values never go stale because attention weights don't update.

This is the single biggest impact contribution possible.

### GPU inference pipeline

The `--features gpu` build currently accelerates only matmul operations via wgpu. A full GPU inference pipeline would keep intermediates in GPU memory between operations (no CPU roundtrips) and batch positions into single dispatches.

### Vocab embedded in checkpoint

Currently the server needs either a training data file or a BPE tokenizer file to reconstruct the vocabulary. Embedding the vocab mapping in the checkpoint would make serving a single-file operation.

## Architecture notes

The forward pass uses the **GPT-J parallel block formulation**:
```
x = x + attn(LN(x)) + FFN(LN(x))
```

This matches the training engine exactly. If you modify the forward pass, the parallel vs sequential distinction matters — the model was trained with parallel blocks.

The attention is **frozen** — weights don't change during training or inference. The FFN uses **dual-maestro Kerr-ODE** with maestro_in and maestro_out (both at dim=16). These are architectural facts, not implementation choices.

## Code of conduct

Be honest about results. If a change makes streaming faster but breaks non-streaming, say so. If GPU inference works on NVIDIA but crashes on AMD, document it. Every honest finding helps.
