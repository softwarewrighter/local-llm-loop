# Performance Analysis — RTX 3090 vs RTX 5060 Ti vs RTX 3060 vs M1 Max vs P40

How fast the local model runs, and how the host platforms (Arch Linux / NVIDIA
RTX 3090, Arch Linux / NVIDIA RTX 3060 12 GB, and macOS / Apple Silicon M1 Max —
plus a *predicted* NVIDIA Tesla P40) compare for this project's workload. The
per-run traces are in [arch-poc-summary.md](arch-poc-summary.md),
[arch-nvidia-3060-poc-summary.md](arch-nvidia-3060-poc-summary.md) and
[mac-poc-summary.md](mac-poc-summary.md); this doc is the apples-to-apples
analysis.

> **What's actually been run:** the project has been run end-to-end on **M1 Max**
> (the macOS POC, plus **measured `llama-bench` + loop + `cargo test`** for
> gpt-oss-20b and Qwen3.6-35B-A3B-MTP — see its section), **RTX 3090** (the Arch
> POC + `llama-bench`), and **RTX 3060 12 GB** (the CPU-offload POC +
> `llama-bench`). The **RTX 5060 Ti 16 GB** has **measured `llama-bench`** numbers
> (see its section + the [5060 Ti POC](nvidia-5060-poc-summary.md)). It has
> **not** been tested on the **P40** at all — that section is a pure prediction.
>
> So: the **M1 Max**, **RTX 3090**, **RTX 3060** and **5060 Ti** figures are
> controlled `llama-bench` measurements; the **P40** figures are an untested
> prediction. Confidence is noted per section.
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

### Coding models on the RTX 3090 — multi-model comparison

The harness is **model-agnostic**, so the 3090 has been used to evaluate several
local coding models against the same greeter loop. All rows are controlled
`llama-bench` runs with identical flags (`-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0
-p 512 -n 128`, llama.cpp b9728). "Loop" = completes the planner→executor→reviewer
loop and produces an **independently-verified** Rust CLI (`cargo test` passes); see
[nvidia-3090-poc-summary.md](nvidia-3090-poc-summary.md) for the full model-search.

| Model | Arch | Size | Prefill `pp512` | Decode `tg128` | Loop? | Greeting quality |
|-------|------|-----:|----------------:|---------------:|:-----:|------------------|
| Qwable-v1 IQ4_XS | MoE 35B-A3B | 18.9 GiB | ~3,605 t/s | ~158 t/s | ✅ | `Hello, X!` |
| **Qwen3-Coder-30B-A3B** Q4_K_M | MoE 30B-A3B | 17.3 GiB | ~4,043 t/s | **~189 t/s** | ✅ | bare name |
| **Gemma-4-26B-A4B** Q4_K_M | MoE 26B-A4B | 15.6 GiB | ~4,209 t/s | ~142 t/s | ✅ | **`Hello, X!`** ✦ |
| **Qwen3.6-35B-A3B** UD-Q4_K_M | MoE 35B-A3B | 21.1 GiB | ~3,365 t/s | ~143 t/s (**MTP 85% accept**) | ✅ | `Hello, X!` |
| **Gemma-4-31B** Q4_K_M (dense) | dense 31B | 17.4 GiB | ~1,217 t/s | ~36 t/s (→~46 w/ E2B draft) | ✅ | `Hello, X!` |
| **Qwen3.6-27B** Q4_K_M (dense) | dense 27B | 15.7 GiB | ~1,320 t/s | ~41 t/s | ✅ | `Hello, X!` (best structure) |
| gpt-oss-20b MXFP4 | MoE 21B-A3B | 11.3 GiB | **~5,652 t/s** | **~205 t/s** | ❌ | — (JSON ctrl-char) |
| Granite-4.1-8B Q4_K_M | hybrid 8B | 5.0 GiB | — | ~111 t/s | ❌ | — (invalid JSON) |

✦ Gemma-4-26B-A4B produced the highest-quality greeter — the proper `Hello, {name}!`
form with correct `--times 0` error handling — while Qwen3-Coder emitted the bare
repeated name. **The six ✅ models are the verified-working set on this box.**

**MTP (multi-token prediction)** is the cleanest spec-decode seen: Qwen3.6-35B-A3B-MTP
self-speculates from a built-in head (`--spec-type draft-mtp`, *no separate draft
model*) at **85% acceptance** vs 44–55% for the external-draft pairs. It's the
preferred speedup mechanism on the bandwidth-limited boxes (3060, M1 Max) where
decode is the bottleneck.

Note the **dense vs MoE decode gap**: Gemma-4-**31B dense** decodes at ~36 t/s (all
30.7B params active per token), ~4–5× slower than the ~3–4B-active MoEs — exactly
the case where speculative decoding pays off. The **Gemma-4-31B + Gemma-4-E2B** pair
(`--spec-type draft-simple`, exact 262144 vocab, ctx 16384 to fit 24 GB) lifts decode
to ~46 t/s (~1.3×) and completes the loop — the one verified *dense spec-decode
coding pair*. Still far slower than an MoE solo, so the MoE coders win on speed;
the dense pair is the option when you specifically want a dense model.

Takeaways:
- **The fast models here are all ~3–4B-active MoE.** Decode is memory-bandwidth-
  bound on the active set, so a "20–35B" model decodes at 140–205 tok/s. gpt-oss-20b
  is fastest on raw throughput but **fails the loop** (it emits a literal control
  character in its JSON envelope — passes tool-calling, fails clean-JSON).
- **Throughput ≠ usability.** Gemma-4 is the *slowest* of the MoEs yet writes the
  *best* code; gpt-oss is the *fastest* yet unusable in the loop. The selection
  metric is "completes the loop with verified, good code," not tok/s.
- **Speculative decoding** still helps *dense* targets (measured 1.3–1.75× in the
  POC) but none of the dense pairs tried beat these MoE-solo runs, and the best
  coders ship MoE-only, so the practical pattern on a 24 GB card is **fast MoE,
  run solo**.

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

## Measured — RTX 5060 Ti 16 GB (Blackwell / FP4)

The 16 GB **RTX 5060 Ti** (GB206 Blackwell, sm_120, 5th-gen tensor cores,
~448 GB/s GDDR7) is **the FP4 box**: it executes **MXFP4 / NVFP4** matmuls
*natively*, where Ampere/Ada upcast them. `llama-bench` with the project's serving
flags (`-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 128`), llama.cpp `3fc4e10`,
speculative decoding off. VRAM is the measured bench-time peak (weights + small KV).
Full write-up: [nvidia-5060-poc-summary.md](nvidia-5060-poc-summary.md).

| Model | Arch | Size | VRAM peak | Prefill `pp512` | Decode `tg128` | Fit |
|-------|------|-----:|----------:|----------------:|---------------:|-----|
| **gpt-oss-20b MXFP4** | MoE 21B-A3B | 11.27 GiB | 11.8 GB | **5,920 t/s** | **138 t/s** | whole (FP4) |
| Qwen3-8B Q4_K_M | dense 8B | 4.68 GiB | 5.2 GB | 3,679 t/s | 81 t/s | whole |
| Phi-4-14B Q4_K_M | dense 14B | 8.28 GiB | 8.9 GB | 2,033 t/s | 46 t/s | whole |
| Qwen3-Coder-30B-A3B Q4_K_M | MoE 30B-A3B | 17.28 GiB | 15.5 GB | 874 t/s | 81 t/s | `--n-cpu-moe 8` (RAM offload) |

> ⚠️ The **loop/gate** results are not yet collected on the 5060 Ti — the
> automation environment kills any persistent `llama-server` (SIGSTKFLT) before it
> can serve, while batch `llama-bench` runs fine. The throughput above is real;
> run the loop from an interactive shell to finish the gates (see the POC summary).

### The FP4 headline: native MXFP4 prefill beats the 3090

Same gpt-oss-20b MXFP4 GGUF, 5060 Ti vs the much pricier 3090:

| gpt-oss-20b MXFP4 | RTX 5060 Ti | RTX 3090 | ratio | bound by |
|---|---:|---:|---:|---|
| Prefill `pp512` | **5,920 t/s** | ~5,652 t/s | **1.05×** | compute (FP4 matmul) |
| Decode `tg128` | 138 t/s | ~205 t/s | 0.67× | bandwidth |
| Bandwidth | ~448 GB/s | ~936 GB/s | 0.48× | — |

- **Prefill — the 5060 Ti wins.** Despite ~⅓ the FP16 tensor throughput and half
  the bandwidth, it *edges out* the 3090 on this model because Blackwell runs the
  MXFP4 matmuls natively while Ampere upcasts them first. Prefill is the
  compute-bound regime agentic coding leans on hardest, so this matters.
- **Decode — bandwidth still wins, but FP4 narrows it.** At 0.67× the 3090 the
  5060 Ti beats its 0.48× bandwidth ratio: native FP4 means fewer effective bytes
  per step. The 3090's 2× bandwidth only buys ~1.5× here.
- **FP4 only helps FP4 models.** The K-quant coders get none of it:
  Qwen3-Coder-30B-A3B **Q4_K_M** decodes 81 t/s here (and must offload experts to
  RAM to fit 16 GB) vs ~189 t/s fully-resident on the 3090. For non-FP4 weights
  the 5060 Ti is a bandwidth-and-offload story like the 3060, not a 3090 rival.

### Qwen3-Coder-30B-A3B doesn't fit 16 GB — the `--n-cpu-moe` curve

The Q4_K_M is 17.28 GiB, so experts spill to the 251 GB system RAM. Fewer offloaded
layers = faster, until VRAM OOMs (**N=8 is the floor that fits**, 15.5/15.8 GB):

| `--n-cpu-moe` | Prefill `pp512` | Decode `tg128` | Fits 16 GB? |
|--------------:|----------------:|---------------:|:-----------:|
| 8 | 874 t/s | 81 t/s | ✅ |
| 12 | 662 t/s | 63 t/s | ✅ |
| 16 | 550 t/s | 55 t/s | ✅ |
| 24 | 401 t/s | 39 t/s | ✅ |
| 4 | — | — | ❌ OOM |

Same shape as the [3060's offload curve](#measured--rtx-3060-12-gb-cpu-expert-offload):
decode survives offload (only ~3.3B active params per token), prefill takes the
hit (RAM-resident experts compute on the CPU). The 5060 Ti's faster GDDR7 +
Blackwell tensor cores keep both regimes well ahead of the 3060's Qwable numbers.

## Measured — M1 Max (Apple Silicon / Metal, 64 GB)

Two of the models the 5060 Ti POC blessed, now **measured** on an **Apple M1 Max
(10-core, 64 GB unified memory, ~400 GB/s, Metal)**, llama.cpp `75ad0b23e`
(b9770), opencode 1.17.9. `llama-bench` with the project's serving flags
(`-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 128`), speculative decoding off —
**directly comparable** to the 3090 / 5060 Ti tables above.

> **The unified-memory advantage.** Both models fit **whole** in the 64 GB
> unified memory — **no expert offload**. The 16 GB 5060 Ti had to spill
> Qwen3-Coder-30B / Gemma-4-26B into system RAM; the Mac never pays that penalty.
> So unlike the 3060/5060, *every* M1 Max row below is a clean full-residency
> measurement, even for the 21 GB MoE.

| Model | Arch | GGUF | Prefill `pp512` | Decode `tg128` | Fit |
|-------|------|-----:|----------------:|---------------:|-----|
| **gpt-oss-20b MXFP4** | MoE 21B-A3B | 11.27 GiB | **998 t/s** | **75.9 t/s** | whole (no FP4 accel) |
| **Qwen3.6-35B-A3B-MTP** UD-Q4_K_M | MoE 35B-A3B | 21.10 GiB | 827 t/s | 47.2 t/s (MTP off) | whole |

### MTP self-speculative decode — a real, measured win on this box

performance-analysis.md predicted MTP would be *the* speedup on bandwidth-bound
Apple Silicon. Measured via `llama-server --spec-type draft-mtp` (built-in
next-token head, **no separate draft model**), same prompt, temp 0, same server —
a clean A/B:

| Qwen3.6-35B-A3B-MTP decode | t/s | draft acceptance |
|---|---:|---:|
| MTP **off** (`SPEC_OFF=1`) | 45.6 | — |
| MTP **on** (`draft-mtp`, n=4) | **56.7** | 179/278 = **64%** |

→ **~1.24× decode** from the embedded MTP head, free of a second model's VRAM.
(The 64% accept here is below the 3090's reported 85% — n=4 / temp 0 / this quant
— but the uplift is real and costs nothing.) `start-qwen36-mtp.sh` ships MTP on.

### Gates + loop (both pass)

Both clear gate 1 (native `tool_calls`) and complete the full
plan→execute→review loop, producing a `greet` crate that **builds and passes
`cargo test` (2/2)** — independently verified, not the model's own claim:

| Model | Gate 1 | Loop | `cargo test` | Run shape |
|-------|:------:|:----:|:------------:|-----------|
| **Qwen3.6-35B-A3B-MTP** | ✅ | ✅ | ✅ 2/2 | clean — 3 steps, all *Continue*, single crate |
| **gpt-oss-20b MXFP4** | ✅ | ✅ | ✅ 2/2 | single crate (one reviewer *Skip*); the `emit` self-heal channel carries its JSON, as on the 5060 |

### Cross-hardware reading — M1 Max vs 5060 Ti vs 3090

Same models, the three measured boxes (decode = `tg128`, prefill = `pp512`,
spec-decode off unless noted):

| | M1 Max (64 GB) | RTX 5060 Ti (16 GB) | RTX 3090 (24 GB) |
|---|---:|---:|---:|
| **gpt-oss-20b** decode | 75.9 | 138 | ~205 |
| **gpt-oss-20b** prefill | 998 | 5,920 | ~5,652 |
| **Qwen3.6-35B-A3B** decode | 47.2 (57 MTP) | — | ~143 (MTP) |
| **Qwen3.6-35B-A3B** prefill | 827 | — | ~3,365 |
| Holds the 21 GB MoE whole? | **yes** (unified) | no (offload) | yes |

Three takeaways:

1. **Decode is the bandwidth story it should be.** gpt-oss-20b at 75.9 t/s is
   ~0.55× the 5060 Ti and ~0.37× the 3090 — close to the bandwidth ratios
   (400 / 448 = 0.89; 400 / 936 = 0.43), with the Mac trailing a little more on
   the FP4 model because **Metal has no native MXFP4 path** (it upcasts; only
   Blackwell accelerates it). Still very usable: ~50–76 t/s solo across both MoEs.
2. **Prefill is where Apple Silicon pays.** 998 / 827 t/s is **~6× under** the
   NVIDIA cards — the M1 Max GPU has no tensor-core matmul unit, exactly the
   compute-bound regime agentic coding leans on hardest (big per-turn prompts).
   This, not decode, is the real cost of running the loop on a Mac.
3. **The Mac's edge is capacity, and MTP.** 64 GB unified holds the 21 GB MoE
   *and* a full 128k KV with room to spare — no offload, no quant-down, and the
   built-in **MTP head buys ~1.24× decode for free** on the bandwidth-bound box
   exactly as predicted. The 5060 Ti wins throughput per dollar; the M1 Max wins
   when the model (or the context) doesn't fit a 16 GB card.

**Net:** for *this* workload the ranking is **3090 > 5060 Ti > M1 Max on raw
speed** (decode ~2–3×, prefill ~6×), but the M1 Max **completes the same loop
with the same verified output** and is the only one of the three that never has
to offload — so it's the comfortable choice when capacity matters more than
turn latency.

### MLX vs GGUF on the M1 Max — Ornith-1.0-35B (2026-06-28)

The first **MLX-runtime** numbers here. Ornith-1.0-35B (Qwen-3.5 MoE, ~3B active)
run two ways on the M1 Max: GGUF via `llama-server` (b9770) and MLX via
`mlx_lm.server` (mlx-lm 0.31.3). Both clear gate 1 (native `tool_calls`). Full
family writeup: [plan-ornith-models.md](plan-ornith-models.md).

| Ornith-35B | Runtime | Decode | Loop → working code | `cargo test` |
|------------|---------|-------:|--------------------:|:------------:|
| **MLX 6-bit** | mlx_lm.server | **55.6 t/s** | **6m06s** (clean, 3 steps) | ✅ 3/3 |
| GGUF Q6_K | llama.cpp | 40.7 t/s | 8m40s (1 wander + 1 Skip) | ✅ 2/2 |

- **MLX wins ~1.35× on decode and ~1.4× on wall-clock-to-working-code** — Apple's
  framework is better tuned for Metal than llama.cpp's generic Metal backend.
  (For reference the 9B-Dense shows the same shape: MLX 8-bit ~37 vs GGUF Q8_0
  ~33 t/s.)
- **Both end green**, so capability is equal; the gap is speed + reliability. The
  GGUF run's one blemish was an off-task file write *outside* the workspace (a
  llama.cpp Qwen-template/tool quirk, caught by the harness's external-dir guard)
  — the same weights ran clean under MLX.
- **Takeaway:** on Apple Silicon, prefer **MLX** for these Qwen-3.5 models; it's
  the faster, cleaner default and `mlx_lm.server` does native tool-calling +
  `--draft-model` speculative decoding. GGUF/llama.cpp remains the portable path
  and the apples-to-apples tie to every other row here.

### Dense + MTP on the M1 Max — Qwopus3.6-27B-Coder (2026-07-02)

The cleanest **dense-vs-MoE** datapoint on this box. `Qwopus3.6-27B-Coder-MTP`
(Qwen-3.6 **dense** 27B, Opus-distilled, **no-think** — 67% SWE-bench off-thinking)
Q6_K via `llama-server`, with its **MTP head** (`--spec-type draft-mtp`). Clears
gate 1; loop completes; `cargo test` **2/2**.

| Qwopus-27B (dense) | Prefill | Decode | Loop → working code |
|--------------------|--------:|-------:|--------------------:|
| MTP **off** (llama-bench) | 125 t/s | 12.3 t/s | — |
| MTP **on** (77% accept) | — | **16.0 t/s** (~1.30×) | **19m43s** (2/2) |

- **MTP works well here** — 77% draft acceptance (the highest measured; vs 64% for
  Qwen3.6-35B-MTP), lifting decode ~1.30×. But it can't overcome the architecture.
- **Dense is the story.** At 16 t/s the loop takes **19m43s** — ~2.5–3× the MoE
  coders on the *same box* (Ornith-35B MLX 6m06s, Qwen3.6-35B-MTP 8m01s), all of
  which decode 3–4× faster because only ~3B params are active per token. This is
  **not** a reasoning-token tax: it's a no-think model (0 `<think>` blocks), so the
  20 min is pure dense decode.
- **Takeaway:** a genuinely strong coder, but **dense ⇒ too slow on Metal**. Its
  better home is a box where it fits whole on a fast GPU — the **RTX 3090** (Q6_K
  is 22.4 GB, fits 24 GB whole; ~4–5× the M1 Max's decode with real tensor cores).
  On the M1 Max the MoE coders remain the practical picks.

> **Projected — RTX 3090 24 GB (untested).** Dense decode is bandwidth-bound, so
> the 3090's ~936 GB/s (**2.3×** the M1 Max) should lift decode from 12.3 →
> **~30–40 t/s** (anchored to the measured **~41 t/s for Qwen3.6-27B dense** on the
> 3090), and MTP's 1.30× → **~40–55 t/s**. Prefill jumps **~10×** (real tensor
> cores vs none) — the regime agentic coding leans on hardest. Net: the loop should
> fall from **19m43s to roughly ~5–8 min** — night-and-day, and why the 3090 is its
> home. **Fit caveat:** Q6_K (22.4 GB) + the q8_0 KV at ctx 32k (~2 GB on a 27B) is
> borderline on 24 GB and may OOM whole-resident — use **Q5_K_M (19.5 GB)** for
> full-context headroom (negligible quality loss). Even so it stays **dense**, so it
> won't beat the A3B MoE coders on the 3090 (Qwen3-Coder-30B ≈ 3m30s whole-resident);
> the 3090 makes Qwopus *practical*, not *fastest*.

### Qwen3.8-27B Q4_K_M on the M1 Max -- correct, sleep-contaminated timing (2026-08-15)

The exact requested text-only Q4_K_M is 15.93 GiB and ran at 32k context with
full Metal offload, q8_0 KV, flash attention, and no vision projector. Process
RSS after the run was 20.5 GiB.

| Metric / gate | Measured result |
|---------------|-----------------|
| Native tool call | ✅ exact `write_file` call |
| Structured plan | ✅ attempt 2; attempt 1 used a rejected absolute path |
| Step/review envelopes | ✅ all five first try |
| Decode | 12.4 t/s at gate; roughly 8-12 t/s while actively generating in-loop |
| Prefill | 114.1 t/s at gate |
| Full loop | ✅ five executed steps; **2h43m57.45s**, including laptop sleep |
| Independent validation | ✅ 3/3 tests, clippy, fmt, repeat and invalid-zero behavior |

Code quality was excellent for the task. Qwen3.8 caught and repaired a clap
parser type mismatch at runtime—the same class of defect Laguna left
uncompiled—then reran its acceptance checks. It added a third zero-repeat unit
test and finished with clean lint and formatting. At least two visible sleep
gaps contaminate wall-clock and aggregate server timing, so this run establishes
fit and quality but is not a clean throughput comparison.

### Off-target Qwen3.5-27B Q4_K_M comparison -- correct but extremely slow (2026-08-15)

The text-only 15.59 GiB Unsloth Q4_K_M was run at 32k context with full Metal
offload, q8_0 KV, flash attention, and no vision projector. This 27B dense hybrid
uses 48 Gated DeltaNet and 16 attention blocks, but all parameters remain active
during decode. The llama-server process showed 24.1 GiB RSS after the run.

| Metric / gate | Measured result |
|---------------|-----------------|
| Native tool call | ✅ exact `write_file` call |
| Structured plan | ✅ first try; valid six-step `emit` envelope |
| Step/review envelopes | ✅ all first try; no harness retries |
| Decode | ~12.1 t/s at gate; ~9-11 t/s in the growing loop |
| Prefill | 104.8 t/s at gate; roughly 100-220 t/s in-loop |
| Full loop | ✅ five executed steps, one skipped; **2h24m21.34s** |
| Independent build | ✅ `cargo test` **2/2**; valid repeat and zero-input behavior |

Code quality was good: a small clap CLI, a pure `greet()` function, correct
newline-separated output, explicit zero validation, and the two requested tests.
Agent efficiency was exceptionally poor. Most phases generated thousands of
reasoning tokens and repeatedly looked for files at the workspace root before
recovering to the nested `greet/` crate. Step 4 completed the remaining coding
work, so its reviewer skipped step 5; the harness still executed the already
satisfied test step. The final `Stop` meant “all work complete,” not a real
intervention request. This is the slowest successful loop in the README table.

### Laguna S 2.1 IQ3_XXS on the M1 Max -- measured failure (2026-08-15)

Laguna S 2.1 (118B-A8B MoE) was tested with the cached 41.24 GiB
`unsloth/Laguna-S-2.1-GGUF` `UD-IQ3_XXS` file. It used Poolside's `laguna`
llama.cpp branch (`04b2b72cb`, build 10008), full Metal offload, one 16k slot,
Jinja, and no speculative decoding. The model loaded in 42 s.

| Metric / gate | Measured result |
|---------------|-----------------|
| Native tool call | ✅ exact file-write smoke test |
| Structured plan | ✅ first try; valid six-step `emit` envelope |
| Step 1 + review | ✅ first try; valid result + Continue review |
| Decode | ~27-29 t/s during long generations |
| Large-prompt prefill | ~230-260 t/s |
| Full loop | ❌ stopped at **13m02.6s**, still in step 2/6 |
| Independent build | ❌ `E0599` from generated clap parser; no tests present |

The failure was agent efficiency and correctness, not model loading or tool-call
parsing. Step 2 wrote a plausible `src/main.rs`, then entered repeated long tool
turns, filled the 16k context once, and stopped changing files. The generated
`clap::value_parser!(usize).range(1..)` did not compile, and the model never ran
a successful build to catch it. This result is deliberately excluded from the
successful wall-clock table. Full trace and artifact details are in the
[August 2026 evaluation plan](plan-august-2026-models.md).

### Nemotron 3.5 Lightning 30B-A3B Q4_0 on the M1 Max -- planner gate failure (2026-08-15)

The official ggml-org Q4_0 is the requested Mamba-heavy architecture: 23 Mamba,
23 MoE, and 6 attention blocks. Its 17.60 GiB base-model file loaded with full
Metal offload at 32k context in about 9.2 s; the separate MTP file was not loaded.

| Metric / gate | Measured result |
|---------------|-----------------|
| Native tool call | ✅ exact file-write smoke test |
| Decode | ~56-62 t/s |
| Large-prompt prefill | ~650-675 t/s |
| Structured plan | ❌ three attempts; no valid `emit` envelope |
| Planner wall-clock | **1m30.43s** to terminal failure |
| Full loop / build | not reached |

Each plan attempt chose an unnecessary filesystem listing: one used `/*`, and
two malformed the external workspace path. OpenCode rejected those calls, after
which the planner exhausted its retries without producing JSON. This is a
planning/protocol failure, not a fit, loading, or native-tool-use failure.

A BF16-derived Unsloth `UD-Q4_K_XL` was additionally rejected by two llama.cpp
builds (`expected 417 tensors, got 408`). The working official distribution
avoids that packaging mismatch by separating the base target and MTP weights.
Full details are in the [August 2026 evaluation plan](plan-august-2026-models.md).

## The two regimes

Single-stream LLM inference has two distinct performance regimes, and they favor
hardware differently:

| Regime | Bound by | Dominates when |
|--------|----------|----------------|
| **Decode** (generation, 1 token at a time) | **memory bandwidth** | long outputs |
| **Prefill** (prompt / context ingestion) | **compute (matmul)** | long prompts — i.e. agentic coding, where each turn re-sends system prompt + tool schemas + file contents |

### Hardware

| | RTX 3090 | RTX 5060 Ti 16 GB | RTX 3060 12 GB | M1 Max | Tesla P40 |
|---|---|---|---|---|---|
| Memory bandwidth | ~936 GB/s | ~448 GB/s | ~360 GB/s (VRAM) + ~60–77 GB/s/socket (system RAM) | ~400 GB/s | ~346 GB/s |
| Native FP4 (MXFP4/NVFP4) | upcast | **accelerated** (5th-gen TC) | upcast | upcast | upcast |
| FP16 **tensor-core** matmul | ~71–142 TFLOPS | ~90 TFLOPS (5th-gen TC) | ~51–102 TFLOPS (only for GPU-held layers) | — (no tensor cores) | — (no tensor cores) |
| INT8 tensor (MMQ path) | ~284 TOPS | ~180 TOPS | ~102 TOPS (GPU layers only) | — | ~47 TOPS (DP4A, no tensor cores) |
| Native FP16 (non-matmul) | full | full | full | full | **1/64 rate (~0.18 TFLOPS) — effectively broken** |
| Fits the 17.6 GiB model? | **yes** (24 GB) | **no** — 16 GB; light expert-offload (but the 11.3 GiB FP4 gpt-oss fits whole) | **no** — experts offloaded to RAM | yes (unified) | yes (24 GB) |
| Max KV cache (this model) | ~32k tokens (24 GB cap) | ~32k (16 GB) | ~32k (12 GB, mostly model+experts) | full 128k (unified memory) | ~32k tokens (24 GB cap) |

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
