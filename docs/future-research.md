# Future research

Investigations deliberately **deferred** — captured here with enough analysis to
pick up later without re-deriving. Nothing here has been downloaded or run.

---

## 1. DeepSeek-V4 / DSpark speculative decoding

[deepseek-ai/DeepSeek-V4-Pro-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark)
and the V4 family. **Status: out of scope today (too large for every box here);
revisit if a small-enough variant + runtime support appears.**

### What DSpark actually is (the key correction)

DSpark is **not** a small standalone draft model you attach to a larger target.
It is a **self-speculation module baked into the V4 checkpoint itself** — DeepSeek's
branded MTP/EAGLE-style head. The config shows `num_nextn_predict_layers: 1` plus
`dspark_block_size`, `dspark_markov_rank`, `dspark_target_layer_ids`,
`dspark_noise_token_id`. So `…-Pro-DSpark` accelerates **V4-Pro**, `…-Flash-DSpark`
accelerates **V4-Flash** — each speeds up *its own* host model, exactly like the
Qwen3.6-MTP head (no separate draft, no "draft + bigger target" pairing).

### The family and what it costs

| Model | Arch | Total / active | Native weights | vocab |
|-------|------|---------------:|---------------:|------:|
| **V4-Pro** | MoE 384e / 6a | 1.6T / 49B | **865 GB** (FP4+FP8) | 129280 |
| **V4-Flash** | MoE 256e / 6a | ~160B / ~10B | **160 GB** | 129280 |

Smallest community quants found (HF): V4-Pro IQ1 ≈ 300 GB+; V4-Flash full **Q2
GGUF ≈ 52 GB** (`0xSero/...162B-GGUF`), Q2 MLX ≈ 102 GB, REAP-expert-pruned Q2 ≈
50 GB.

### Fit + why deferred

| Box | V4-Pro | V4-Flash |
|-----|--------|----------|
| M1 Max 64 GB | ❌ 865 GB | ⚠️ only the ~52 GB Q2 GGUF fits whole (tight, 2-bit quality) |
| RTX 3090 24 GB | ❌ | ❌ 52 GB → heavy `--n-cpu-moe` RAM offload |
| 3060 12 / 5060 16 GB | ❌ | ❌ far over |

**Two blockers, not just size:**
1. **No runtime support for DSpark.** It needs DeepSeek's custom **DeepSpec**
   runtime; neither `llama.cpp` nor `mlx_lm` parses the `deepseek_v4` DSpark/nextn
   head today (llama.cpp's DeepSeek-MTP support has only ever been partial). Most
   quantized GGUF/MLX builds strip the head entirely — so even the ~52 GB Q2 Flash
   that *fits* the M1 Max would run **without** DSpark speculation.
2. **V4 is a datacenter model.** Pro wants a multi-GPU / huge-RAM node; Flash is
   borderline-runnable only as 2-bit on the M1 Max, at low quality.

### Revisit when
- A `deepseek_v4` + DSpark/nextn path lands in `llama.cpp` or `mlx_lm`, **and**
- a quant that preserves the head fits a target box.

Then the test is: does DSpark self-speculation give the same kind of ~1.2–1.85×
decode uplift we measured for Qwen3.6-MTP? Until both hold, **stick with the
validated self-spec/draft options** (Qwen3.6-35B-MTP, Ornith MLX `--draft-model`).

---

## 2. Speculative decoding with an *offloaded* target — RTX 3090 + Xeon

**Question (for later):** the RTX 3090 box has 24 GB VRAM **plus many Xeon cores
and large system RAM**. For a target model too big for 24 GB (experts offloaded to
RAM via `--n-cpu-moe`), is pairing it with a small GPU-resident draft worth it?

**Short answer: yes, worth a bounded experiment — most clearly for a *dense*
offloaded target; smaller/marginal for MoE, where built-in MTP usually wins.**

### Why it *should* help (the mechanism)

Offloaded decode is slow for two reasons: (1) **RAM bandwidth** to read
CPU-resident expert/layer weights every token (~60–77 GB/s per DDR4 socket vs
~936 GB/s VRAM), and (2) **CPU matmul** for those weights (AVX2, no AMX on
E5-v4-class parts). Speculative decoding replaces *N sequential single-token
decodes* with: draft proposes **K** tokens (cheap, on the GPU), then the target
**verifies all K in one forward pass**.

That verify pass reads each weight **once and reuses it across the K positions**.
For the RAM-resident weights — the expensive part — this **amortizes the slow RAM
read over K tokens**. If decode is bandwidth-bound (it usually is under offload),
that is exactly the bottleneck spec-decode attacks. The draft, being small and
GPU-resident, costs almost nothing and doesn't compete for RAM bandwidth.

### Why the win is smaller than full-GPU spec-decode

- **MoE expert divergence.** Across K speculative positions, different experts may
  fire, so the batched verify touches *more* distinct experts than a single token
  → less weight-reuse amortization. **Dense targets amortize perfectly; sparse MoE
  only partially.**
- **Verify is CPU-compute-heavier.** K positions × expert matmuls on AVX2. If the
  offloaded portion is *compute*-bound rather than bandwidth-bound, batching does
  K× the CPU FLOPs and can erase the bandwidth saving. **The Xeon's many cores are
  the key asset here** — more parallel verify throughput tilts this favorably.
- **MTP makes external drafts redundant.** Where the target ships a co-trained
  head (Qwen3.6-MTP, GLM, DeepSeek-DSpark), self-speculation needs **no draft, no
  extra VRAM**, and accepts at a higher rate. Prefer it; reserve external drafts
  for targets *without* a head.
- **Prefill is untouched.** Spec-decode only speeds decode. Offloaded *prefill*
  (CPU matmul over the whole prompt) stays the dominant turn-latency cost for
  agentic coding — so a decode win doesn't fix the slow turns, it just helps long
  generations.

### Draft choice (same-vocab is mandatory)
- Small (0.5–3B), **exact-vocab**, GPU-resident: e.g. Qwen3-0.6B for Qwen3 targets
  (151936 vocab), Gemma-4-E2B for Gemma. It must fit the VRAM left after the
  target's GPU-held attention/dense layers.
- Or the built-in MTP head where present (preferred).

### The experiment to run (later)
1. **Baseline:** offloaded target solo decode t/s at a chosen `--n-cpu-moe N`.
2. **+ external draft** (GPU-resident, exact vocab): decode t/s, **draft
   acceptance**, net speedup.
3. **+ MTP self-spec** where the target has a head — compare to the external draft.
4. **Sweep** `--n-cpu-moe N` (offload fraction) × draft length K — find the knee.
5. **Dense vs MoE target** — expect dense to benefit clearly more.
6. Record **prefill separately** (it won't improve; set expectations on turn time).

### Hypothesis / verdict to confirm
- **Dense target that overflows 24 GB** (e.g. a 70B dense at `--n-cpu-moe`) + a
  small GPU draft → spec-decode should recover a real fraction of the decode loss
  from offload. **The strongest "worth it" case.**
- **Big MoE offloaded** → decode is already cheap-per-token (only active params
  read), and expert divergence dilutes the verify amortization → smaller gain;
  built-in MTP usually beats an external draft. **Marginal — measure before
  investing.**
- The many Xeon cores materially help the CPU-side batched verify, which is the
  main risk factor. Net: **a bounded, well-instrumented A/B is worth running**,
  with calibrated expectations (decode-only win; prefill unchanged).

Related: the GPU-helps-prefill premise in [glm-models.md](glm-models.md) and the
measured offload curves in [performance-analysis.md](performance-analysis.md)
(RTX 3060, RTX 5060 Ti `--n-cpu-moe`).
