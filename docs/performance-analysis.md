# Performance Analysis — Qwable-v1 IQ4_XS, RTX 3090 vs RTX 3060 vs M1 Max vs P40

How fast the local model runs, and how the host platforms (Arch Linux / NVIDIA
RTX 3090, Arch Linux / NVIDIA RTX 3060 12 GB, and macOS / Apple Silicon M1 Max —
plus a *predicted* NVIDIA Tesla P40) compare for this project's workload. The
per-run traces are in [arch-poc-summary.md](arch-poc-summary.md),
[arch-nvidia-3060-poc-summary.md](arch-nvidia-3060-poc-summary.md) and
[mac-poc-summary.md](mac-poc-summary.md); this doc is the apples-to-apples
analysis.

> **What's actually been run:** the project has been run end-to-end on **M1 Max**
> (the macOS POC), **RTX 3090** (the Arch POC + `llama-bench`), and **RTX 3060
> 12 GB** (the CPU-offload POC + `llama-bench`). It has **not** been tested on the
> **P40** at all — that section is a pure prediction.
>
> So: the **RTX 3090** and **RTX 3060** figures are controlled `llama-bench`
> measurements; the **M1 Max** figures are derived from its POC run plus hardware
> characteristics; the **P40** figures are an untested prediction. Confidence is
> noted per section.
>
> ⚠️ **The 3060 is a different regime from everything else here.** Every other row
> holds the **whole model in the accelerator's memory**. The 3060 has only 12 GB —
> too small for the 17.6 GiB model — so its experts live in **system RAM** and only
> part of the model is GPU-resident. Its numbers are *not* a like-for-like GPU
> comparison; they measure a hybrid GPU + CPU/RAM setup (the same regime as
> [GLM-5.2](glm-models.md)). See its own section below.

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

## Measured — RTX 3060 12 GB (CPU expert-offload)

`llama-bench` on the 3060 with the project's serving flags, **but the model does
not fit in 12 GB** — so the MoE experts are pushed to system RAM with
`--n-cpu-moe N` (experts of the first N of 40 layers → CPU). Two operating points,
build **b9784** (`8be759e`), speculative decoding off:

```bash
# tuned: 22 of 40 expert layers on the GPU (~11.2 GB VRAM)
llama-bench -m Qwable-v1.IQ4_XS.gguf -ngl 99 --n-cpu-moe 18 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 128
# max offload: all experts in RAM
llama-bench -m Qwable-v1.IQ4_XS.gguf -ngl 99 --n-cpu-moe 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 128
```

| Metric | 3060 — all experts in RAM | 3060 — tuned (22/40 on GPU) |
|--------|--------------------------:|----------------------------:|
| Prompt / prefill (`pp512`) | ~234 tok/s | **~413 tok/s** |
| Generation (`tg128`) | ~33 tok/s | **~49 tok/s** |

Host: Xeon E5-2697 v4 (2×18c, quad-channel DDR4 per socket), 503 GB RAM, CUDA 13.

**Reading these numbers:**

- **`--n-cpu-moe` is the throttle.** Moving expert layers from RAM onto the GPU
  lifts *both* regimes (prefill +77%, decode +48% going from all-CPU to 22-on-GPU)
  because those layers' matmuls then run on the GPU instead of the CPU, and their
  weights are read from VRAM (~360 GB/s) instead of system RAM (~60–77 GB/s per
  socket). Smaller N = faster, until the 12 GB OOMs. N=18 fills ~11.2 GB here.
- **Decode (~49 tok/s)** is bounded by RAM bandwidth for the CPU-resident experts.
  It lands ~3× under the 3090 and just below the M1 Max — whose unified memory
  holds the *whole* model, so none of its experts pay the system-RAM penalty.
- **Prefill (~413 tok/s)** takes the bigger hit — ~9× under the 3090. Prefill is
  compute-bound; the experts in RAM are computed on the CPU (AVX2, no AMX/tensor
  cores), so the 3060's tensor cores only accelerate the layers they hold. This is
  the structural cost of not fitting in VRAM, and it's the regime agentic coding
  leans on hardest (big per-turn prompts).

> Why decode survives offload but prefill suffers: decode reads only the **~3B
> active** params per token (a bandwidth problem RAM can mostly keep up with),
> while prefill does **dense matmuls over the whole prompt** (a compute problem the
> CPU is bad at). Same reason the [GLM-5.2](glm-models.md) plan attaches a GPU
> mainly to help *prefill*.

## The two regimes

Single-stream LLM inference has two distinct performance regimes, and they favor
hardware differently:

| Regime | Bound by | Dominates when |
|--------|----------|----------------|
| **Decode** (generation, 1 token at a time) | **memory bandwidth** | long outputs |
| **Prefill** (prompt / context ingestion) | **compute (matmul)** | long prompts — i.e. agentic coding, where each turn re-sends system prompt + tool schemas + file contents |

### Hardware

| | RTX 3090 | RTX 3060 12 GB | M1 Max | Tesla P40 |
|---|---|---|---|---|
| Memory bandwidth | ~936 GB/s | ~360 GB/s (VRAM) + ~60–77 GB/s/socket (system RAM) | ~400 GB/s | ~346 GB/s |
| FP16 **tensor-core** matmul | ~71–142 TFLOPS | ~51–102 TFLOPS (only for GPU-held layers) | — (no tensor cores) | — (no tensor cores) |
| INT8 tensor (MMQ path) | ~284 TOPS | ~102 TOPS (GPU layers only) | — | ~47 TOPS (DP4A, no tensor cores) |
| Native FP16 (non-matmul) | full | full | full | **1/64 rate (~0.18 TFLOPS) — effectively broken** |
| Fits the 17.6 GiB model? | **yes** (24 GB) | **no** — experts offloaded to RAM | yes (unified) | yes (24 GB) |
| Max KV cache (this model) | ~32k tokens (24 GB cap) | ~32k (12 GB, mostly model+experts) | full 128k (unified memory) | ~32k tokens (24 GB cap) |

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

## Predicted — NVIDIA Tesla P40 (24 GB, Arch Linux)

> ⚠️ **Untested.** The project has not been run on a P40. Everything below is a
> prediction from hardware specs and the measured 3090 baseline — no P40 numbers
> have been observed. Treat as a hypothesis to verify, not a result.

The P40 (Pascal GP102, 2016) is a peculiar card: **crippled FP16 but fast INT8**.
That one fact drives everything for llama.cpp. It has no tensor cores, and its
native FP16 runs at **1/64 of FP32** (~0.18 TFLOPS) — so any FP16 compute path is
catastrophic. It survives only because llama.cpp's **MMQ kernels use Pascal's
DP4A INT8** instructions (~47 TOPS), which Pascal does at full speed. With 24 GB
GDDR5 it holds the same ~32k context as the 3090.

**Decode / generation (bandwidth-bound — higher confidence).** Scaling the
measured 3090 figure (158 tok/s) by the bandwidth ratio:

- 158 × (346 / 936) ≈ **~58 tok/s** ceiling → realistically **~45–55 tok/s** after
  Pascal-era memory-subsystem and kernel inefficiencies.
- That lands the **P40 just *below* the M1 Max** (~65–70) — same league, M1 Max a
  bit ahead on raw bandwidth — and both at roughly **⅓ of the 3090**.

**Prefill / prompt processing (compute-bound — low confidence).** The P40 can't
use the FP16 path the 3090 leans on; it rides the **DP4A INT8 MMQ path (~47
TOPS)** instead. Estimate **~400–700 tok/s** — likely **comparable to or a bit
faster than the M1 Max** (Apple's GPU is ~250–400 tok/s here: decent bandwidth but
no matmul accelerator and a heavier i-quant dequant cost on shader ALUs), yet
still **~5–8× slower than the 3090's ~3,605 tok/s**, which has real tensor cores.

| Metric | RTX 3090 (measured) | M1 Max (pred.) | Tesla P40 (pred.) |
|--------|--------------------:|---------------:|------------------:|
| Prefill (`pp512`) | ~3,605 tok/s | ~250–400 tok/s | ~400–700 tok/s |
| Generation (`tg128`) | ~158 tok/s | ~65–70 tok/s | ~45–55 tok/s |
| Max context (24 GB / unified) | ~32k | 128k | ~32k |

**Net:** for *this* workload the P40 ≈ M1 Max, trading blows — the M1 Max edges
**decode**, the P40 likely edges **prefill** (the regime agentic coding leans on),
so the P40 may feel slightly snappier turn-to-turn while the Mac streams tokens a
touch faster. Both are far behind the 3090; the M1 Max keeps the context-size
crown.

### P40 gotchas (these dominate the prediction)

1. **Keep it on the INT8 path.** If llama.cpp ever falls back to FP16 cuBLAS on a
   P40, throughput collapses 1–2 orders of magnitude (that 1/64 FP16 rate). Modern
   builds pick MMQ/DP4A automatically for quantized models, so defaults are usually
   fine — but this is the #1 cause of "my P40 is unusably slow."
2. **Flash attention is weak on Pascal.** The project's `--flash-attn on` wants
   FP16; on a P40 the FA path is often *slower* than non-FA (or falls back).
   Benchmark `-fa 0` vs `-fa 1` — the answer may flip versus the 3090, especially
   as context grows.
3. **i-quant dequant hurts more** without tensor cores to hide it. A simpler
   k-quant (e.g. `Q4_K_M`) may actually run *better* on a P40 despite being larger.
4. **Older toolchain features** — the CUDA-graph reuse that helps the 3090
   (`graphs reused = 14776` in the run logs) and some newer kernels are less
   effective or absent on compute capability 6.1.
5. **Practical:** passively-cooled 250 W datacenter card — needs a blower shroud,
   and has no video output.

**Confidence:** decode is anchored to a real measurement scaled by bandwidth
(±~20%); prefill is an educated estimate (±~2×) because it hinges on the kernel
path llama.cpp picks for IQ4_XS on Pascal and on the FA setting. Settle it by
running the same `llama-bench` below on the P40 with `-fa` toggled both ways.

## Conclusion

For this project's workload the **RTX 3090 is clearly faster**:

- **~2.3× on generation** — bandwidth-bound, quant-independent, the solid number.
- **Several× on prompt processing** — compute-bound, and the regime agentic
  coding leans on hardest (big per-turn prompts). The 3090's tensor cores have no
  counterpart in the M1 Max GPU.

The M1 Max's one real advantage is **unified memory**: it can hold the full
**128k-token** KV cache, while the 24 GB 3090 is capped at the **32k** context
configured here.

And the **RTX 3060 12 GB** shows the floor: when the model *doesn't fit*, the loop
still runs correctly (identical plan/execute/review behavior and output quality —
see [arch-nvidia-3060-poc-summary.md](arch-nvidia-3060-poc-summary.md)) at
**~49 tok/s decode / ~413 tok/s prefill** with experts in RAM. Decode stays in the
M1 Max's league because only ~3B params are read per token; prefill drops ~9×
below the 3090 because the RAM-resident experts are computed on the CPU. That
trade — cheap card + lots of RAM, GPU accelerating only what it holds — is the
whole premise of the [GLM-5.2](glm-models.md) plan, here validated on a 35B MoE.

### Getting a true number

The prefill multiplier depends on which kernel llama.cpp selects for IQ4_XS (INT8
MMQ vs FP16 dequant), so for a defensible figure run the identical bench on the
M1 Max and compare the `pp512` / `tg128` rows directly:

```bash
HF_HUB_OFFLINE=1 llama-bench -m /path/to/Qwable-v1.IQ4_XS.gguf \
  -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 128
```

Same model, same flags, speculative decoding off on both → directly comparable.
