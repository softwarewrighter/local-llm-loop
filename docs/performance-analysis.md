# Performance Analysis — Qwable-v1 IQ4_XS, RTX 3090 vs M1 Max

How fast the local model runs, and how the two host platforms (Arch Linux /
NVIDIA RTX 3090 and macOS / Apple Silicon M1 Max) compare for this project's
workload. The per-run traces are in [arch-poc-summary.md](arch-poc-summary.md)
and [mac-poc-summary.md](mac-poc-summary.md); this doc is the apples-to-apples
analysis.

## Measured baseline — RTX 3090

`llama-bench` with speculative decoding off and the project's serving flags
(full GPU offload, flash attention, q8_0 KV cache):

```bash
llama-bench -m Qwable-v1.IQ4_XS.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 128
```

| Metric | RTX 3090 |
|--------|----------|
| Prompt / prefill (`pp512`) | **~3,605 tok/s** |
| Generation (`tg128`) | **~158 tok/s** |

(llama.cpp build 9728 / 96399f2.)

> The model reports as `qwen35moe 35B.A3B IQ4_XS` — a **34.7B-total
> mixture-of-experts with ~3B active parameters per token**. Only the active
> experts are read per token, which is why a "35B" model decodes this fast.

### Why the POC numbers (~132 vs ~31 tok/s) are *not* a fair comparison

The Arch POC observed ~132 tok/s generation; the macOS POC reported ~31 tok/s.
That ~4× gap is misleading: the **macOS run had speculative decoding on, the Arch
run did not**. Speculative decoding is a generation-speed multiplier (when draft
acceptance is high), so the two figures measure different things. The clean
comparison is `llama-bench` with the same settings and speculative decoding off
on both machines.

## The two regimes

Single-stream LLM inference has two distinct performance regimes, and they favor
hardware differently:

| Regime | Bound by | Dominates when |
|--------|----------|----------------|
| **Decode** (generation, 1 token at a time) | **memory bandwidth** | long outputs |
| **Prefill** (prompt / context ingestion) | **compute (matmul)** | long prompts — i.e. agentic coding, where each turn re-sends system prompt + tool schemas + file contents |

### Hardware

| | RTX 3090 | M1 Max |
|---|---|---|
| Memory bandwidth | ~936 GB/s | ~400 GB/s |
| FP16 **tensor-core** matmul | ~71–142 TFLOPS | — (no tensor cores) |
| INT8 tensor (MMQ path) | ~284 TOPS | — |
| GPU shader FP16/FP32 | (fallback) | ~10–20 TFLOPS |
| Max KV cache (this model) | ~32k tokens (24 GB cap) | full 128k (unified memory) |

## How the IQ4_XS quantization affects the comparison

A common trap is to reason about a 4-bit model with "INT4 throughput." That's
wrong for this stack:

**Quantization is a storage/bandwidth optimization, not a compute-precision
change.** The 4-bit weights are **upcast before every matmul** — there is no
4-bit arithmetic in the GEMM:

- **CUDA (3090):** runs as MMQ kernels (weights → **INT8**, multiplied on **INT8
  tensor cores** via `dp4a`/`mma.s8`) or dequant → **FP16** cuBLAS on FP16 tensor
  cores. Either way it uses **tensor-core matmul**.
- **Metal (M1 Max):** dequant → FP16/FP32 on the **shader ALUs**. The Apple GPU
  has no tensor-core / matmul accelerator reachable from this path.

So the right compute ceiling is **tensor-core matmul throughput** (expressible in
FP16 or INT8) — *not* INT4 TOPS, since that hardware path isn't used.

Consequences per regime:

- **Decode (bandwidth-bound).** The quant is *exactly why* decode is fast: ~0.53
  bytes/param read instead of 2 (FP16). Both platforms gain proportionally, so
  the **ratio stays ≈ the bandwidth ratio ≈ 2.3×** (936 / 400) and is essentially
  quant-independent. From the 3090's measured 158 tok/s, this predicts the
  **M1 Max around ~65–70 tok/s**.
- **Prefill (compute-bound).** Cheaper weight reads push prefill *further* into
  compute-bound territory (fewer bytes per FLOP), so tensor-core throughput
  matters *more*, not less. The 3090 has dedicated matmul silicon the M1 Max GPU
  lacks, so its prefill advantage is **several×** and larger than a naive FP16
  TFLOPS-only estimate suggests.

**i-quant caveat (small, in the 3090's favor):** IQ4_XS is an *i-quant* with
heavier dequantization (codebook lookups) than k-quants. Both machines pay it;
on the 3090 it's hidden behind the tensor cores, on the M1 Max it competes for
the same shader ALUs doing the matmul — so it costs the Mac relatively more.

## Conclusion

For this project's workload the **RTX 3090 is clearly faster**:

- **~2.3× on generation** — bandwidth-bound, quant-independent, the solid number.
- **Several× on prompt processing** — compute-bound, and the regime agentic
  coding leans on hardest (big per-turn prompts). The 3090's tensor cores have no
  counterpart in the M1 Max GPU.

The M1 Max's one real advantage is **unified memory**: it can hold the full
**128k-token** KV cache, while the 24 GB 3090 is capped at the **32k** context
configured here.

### Getting a true number

The prefill multiplier depends on which kernel llama.cpp selects for IQ4_XS (INT8
MMQ vs FP16 dequant), so for a defensible figure run the identical bench on the
M1 Max and compare the `pp512` / `tg128` rows directly:

```bash
HF_HUB_OFFLINE=1 llama-bench -m /path/to/Qwable-v1.IQ4_XS.gguf \
  -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 128
```

Same model, same flags, speculative decoding off on both → directly comparable.
