# Plan — small-coder testing on an RTX 3060-12G node

How to test **small coding models for this harness** on an **RTX 3060 12 GB**
node, **with and without speculative decoding**. This is a *plan*; it builds on
work already done on this exact class of box — the [12 GB RTX 3060 CPU-offload
POC](arch-nvidia-3060-poc-summary.md) (Qwable with MoE experts in RAM) — and reuses
the method + rules from the [24 GB RTX 3090 model search](nvidia-3090-poc-summary.md)
(the two gates). Start-script templates: [config/arch-nvidia-3060/](config/arch-nvidia-3060/)
and [config/nvidia-3090/](config/nvidia-3090/).

## Hardware profile

| | RTX 3060-12G (the dual-Xeon box) |
|---|---|
| Arch | **Ampere** (GA106), sm_86, 3rd-gen tensor cores |
| VRAM | **12 GB GDDR6**, ~360 GB/s |
| Host RAM | **503 GB** DDR4 (dual Xeon E5-2697 v4) — enables MoE expert-offload |
| FP4 | **not** hardware-accelerated (Ampere upcasts FP4/MXFP4) → prefer K-quants |

**Build:** already have a working CUDA build (sm_86, b9784+) from the 3060 POC.
The same flags apply; confirm `--spec-type` / `--model-draft` exist on the build.

**The 3060's angle:** small VRAM but **huge system RAM**. Two regimes:
- **Small models that fit 12 GB** → fully GPU-resident, fast.
- **MoE models too big for 12 GB** → keep experts in the 503 GB RAM
  (`--n-cpu-moe N`), dense/attention on the GPU — the proven Qwable-on-3060 trick.
  Decode survives (only ~active params read), **prefill takes the hit**.

## VRAM budget (12 GB)

Reserve ~1.5 GB for KV (q8_0) + compute → **~10–10.5 GB for weights** fully
resident, or **target + draft combined ≤ ~10 GB** for speculative decoding. Bigger
MoEs go partly to RAM via `--n-cpu-moe`.

## Candidate models

### A. Solo, fully GPU-resident (≤ ~10.5 GB)

| Model | Type | ~Q4 size | Notes |
|---|---|---:|---|
| **Phi-4-14B** | dense | ~9 GB | Fits (tight w/ ctx); strong structured-output. Best "real coder that fits". |
| **Qwen3-8B** | dense | ~5 GB | Tool-calling OK; small → JSON-gate risk. |
| **Qwen2.5-Coder-7B** | dense | ~4.5 GB | Best small *coder*, **but tool-calling broken in llama.cpp** → speed test only. |
| **CodeGemma-7B / OpenCoder-8B** | dense | ~5 GB | Coder alternatives; verify gates. |
| **Gemma-4-E4B** | dense (small) | ~4 GB | Gemma 4 tools + structured JSON; comfortable. |
| Qwen3-4B | dense | ~2.5 GB | Very small; loop-completion unlikely (gate 2) but a fast draft/inline model. |
| gpt-oss-20b MXFP4 | MoE | ~11.3 GB | **Borderline** — only with small ctx; **slow on Ampere** (FP4 upcast). Prefer the 5060 for this one. |

### B. MoE via expert-offload to the 503 GB RAM (`--n-cpu-moe`)

The bigger/better MoE coders don't fit 12 GB but run with experts in RAM (slower
prefill, decode mostly survives). Same pattern as the 3060 Qwable POC.

| Model | ~Q4 size | Notes |
|---|---:|---|
| **DeepSeek-Coder-V2-Lite-16B** | ~10 GB | MoE ~2.4B active; may *just* fit or light offload. |
| Gemma-4-26B-A4B | ~15.6 GB | tune `--n-cpu-moe` until ~11 GB VRAM (cf. 3060 POC's N=18). |
| Qwen3-Coder-30B-A3B | ~17.3 GB | heavier offload; verify it still completes the loop. |

### C. Speculative-decoding pairs (target + draft both ≤ ~10 GB)

`--spec-type draft-simple` required; vocab must match.

| Target + draft | Combined | Vocab | Notes |
|---|---:|---|---|
| **Qwen3-8B + Qwen3-0.6B** | ~6 GB | exact (151936) | clean, fits comfortably |
| **Qwen3-4B + Qwen3-0.6B** | ~3.5 GB | exact | very light; good for speed studies |
| Qwen2.5-Coder-7B + 0.5B | ~5 GB | 128-gap (tolerated) | high acceptance, **tools broken** → speed test only |

## The two gates + the 3060 reality

Same gates as everywhere: **native tool-calling** + **strict JSON**. On 12 GB you
are pushed toward **smaller** models, which **fail gate 2 (strict JSON) more often**
— so this node is best framed as:
- **Speed/feasibility lab:** measure spec-decode speedups and MoE-offload tradeoffs
  on small pairs (any model, gates irrelevant for raw `llama-bench`).
- **Loop-completion:** expect only the stronger fitters (Phi-4-14B, maybe Qwen3-8B,
  or an offloaded MoE coder) to actually finish with passing `cargo test`.
Higher-precision quants (Q5/Q8 of a 7-8B) are the cheapest way to win gate 2.

## Test procedure

1. Reuse the 3060 CUDA build; confirm spec-decode flags.
2. Per model: start script with its own `--alias`, GGUF path, `PORT 8080`,
   `--cache-type-k/v q8_0 --flash-attn on --jinja`; for big MoEs add
   `--gpu-layers 99 --n-cpu-moe N` (tune N to ~11 GB VRAM).
3. opencode.json entry per alias (zero `cost`; opencode ≥ 1.17).
4. Gate test (native `tool_calls`?) → loop (`MODEL=llamacpp/<alias>
   ./scripts/demo-orchestrate.sh`) → independent `cargo test`.
5. Spec-decode A/B and `--n-cpu-moe` sweep; record decode/prefill, acceptance,
   loop wall-clock, VRAM.

## What to measure / open questions

- For offloaded MoEs: the `--n-cpu-moe` sweep (prefill vs VRAM) — reuse the 3060
  POC's curve (N=18 → ~11.2 GB, ~413 t/s prefill / ~49 t/s decode for Qwable).
- Smallest model that still **completes the loop** with passing tests on 12 GB.
- Does spec-decode help here (Ampere, 360 GB/s) more than on the 3090? (Decode is
  more bandwidth-bound on the slower card → drafts may pay off more.)
