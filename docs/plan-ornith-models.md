# Ornith-1.0 — model family, GGUF/MLX ecosystem, and per-system plan

[Ornith-1.0](https://deep-reinforce.com/ornith.html) (deepreinforce-ai) is a
self-improving open-source **family for agentic coding** — post-trained with RL
that optimizes *both* the solution rollout and the scaffold that drives it. This
doc is the research + deployment plan for running Ornith on this project's harness
across the four test boxes (RTX 3060 12 GB, RTX 5060 Ti 16 GB, RTX 3090 24 GB,
M1 Max 64 GB). Companion: [performance-analysis.md](performance-analysis.md).

- **Date:** 2026-06-28
- **License:** MIT (all sizes)
- **Context:** 262,144 tokens · **Reasoning:** built-in `<think>…</think>` blocks
- **Modality:** vision-language (configs carry `vision_config`; GGUF/MLX ship an
  optional `mmproj`). Irrelevant for coding — don't load the mmproj.

## The four models (2 dense + 2 MoE)

Confirmed from each repo's `config.json` (not the marketing copy). **All three
released sizes are Qwen-3.5-based and share one custom vocab (248320** — *not* the
usual Qwen 151936, which matters for drafts; see [Speculative decoding](#speculative-decoding)).

| # | Model | Arch | Base | Key specs | GGUF? |
|---|-------|------|------|-----------|:-----:|
| 1 | **Ornith-1.0-9B** | **Dense** | **Qwen 3.5** | hidden 4096, vocab 248320 | ✅ |
| 2 | **Ornith-1.0-31B** | **Dense** | **Gemma 4** | — | ❌ **never released** (no repo anywhere) |
| 3 | **Ornith-1.0-35B** | **MoE** | **Qwen 3.5** | 256 experts / **8 active (~3B)**, hidden 2048, vocab 248320 | ✅ |
| 4 | **Ornith-1.0-397B** | **MoE** | **Qwen 3.5** | 512 experts / 10 active, vocab 248320 | ✅ (bartowski, split) |

> **The Gemma catch.** The marketing line is "post-trained on top of Gemma 4 **and**
> Qwen 3.5," but the *only* Gemma-based model is the **31B-Dense**, and it has **no
> GGUF, MLX, or any other community conversion** — searched all 182 Ornith repos on
> HF. In practice **every testable Ornith today is Qwen-3.5-based.**

### Reported quality (from the 9B model card)

The 9B punches far above its size — beating Qwen3.5-**35B** on several agentic
benchmarks:

| Benchmark | Ornith-9B | Qwen3.5-9B | Qwen3.5-35B | Gemma4-31B |
|-----------|----------:|-----------:|------------:|-----------:|
| Terminal-Bench 2.1 (Terminus-2) | **43.1** | 21.3 | 41.4 | 42.1 |
| SWE-bench Verified | **69.4** | 53.2 | 70.0 | 52.0 |
| SWE-bench Pro | **42.9** | 31.3 | 44.6 | 35.7 |

So the small-card 9B is a genuine coder, not a toy.

## Who owns what — the HF ecosystem (182 repos)

Three tiers, then a long community tail:

| Tier | Owner(s) | What |
|------|----------|------|
| **Source** | `deepreinforce-ai` | base weights (9B, 35B, 397B) + official FP8 + official GGUF (9B, 35B only) |
| **Reference GGUF** | `bartowski` | faithful GGUF of all three Qwen sizes incl. **397B** (split) + imatrix |
| **MTP grafts** | `dan9070`, `wang-yang`, `protoLabsAI`, `giaki3003`, `s-batman`, … | community next-token-head builds for self-speculation (GGUF **and** MLX) |
| **Everything else** | `mlx-community`, `OsaurusAI`, `pipenetwork`, … | MLX, FP8, AWQ, EXL3, NVFP4, MXFP4 — 60+ converters |

**MTP is a technique, not an owner** — several people have grafted next-token
heads onto the 35B and 9B in multiple formats.

### Formats and runtimes

| Format | Best repos | Runtime | Notes |
|--------|-----------|---------|-------|
| **GGUF** | `bartowski/...`, `deepreinforce-ai/...-GGUF` | `llama-server` (llama.cpp) | the harness default |
| **MLX** | **`mlx-community/Ornith-1.0-{9B,35B}-*bit`** | `mlx_lm.server` / Ollama MLX | **fastest on Apple Silicon** |
| **GGUF + MTP** | `dan9070/...35B-APEX-MTP-GGUF` (26.2 GB) | `llama-server --spec-type draft-mtp` | self-spec, no draft model |
| **MLX + MTP** | `wang-yang/Ornith-1.0-35B-MTPLX`, `giaki3003/...9B-MTP-MLX-Serve` | `mlx_lm` | self-spec on Mac |

opencode talks to either runtime via its OpenAI-compatible endpoint, exactly like
the existing `llamacpp` provider — add an `mlx` provider pointing at
`mlx_lm.server` (default `:8080`) for the MLX side.

## GGUF quant sizes (for fit math)

**9B-Dense:** Q4_K_M 5.63 · Q5_K_M 6.47 · Q6_K 7.36 · Q8_0 9.53 · bf16 17.92 GB
**35B-MoE:** Q4_K_M 21.17 · Q5_K_M 24.73 · Q6_K 28.51 · Q8_0 36.90 · bf16 69.38 GB · **APEX-MTP 26.23**
**397B-MoE:** smallest is IQ1_S ≈ **81.8 GB** (3-part) → Q4_K_M ≈ 242 GB. **Fits none of the four boxes** (over even the M1 Max's 64 GB).

## Speculative decoding

Ornith offers **three** routes; only the MTP one is worth it on this harness:

1. **External draft within the family.** Shared 248320 vocab means 9B *could*
   draft 35B/397B. **But it's counterproductive:** the 9B is *dense* (all 9B active
   per token), so it decodes **slower** than the 35B-MoE's ~3B-active target. A
   draft must be much *faster* than the target — this is the opposite. ❌ No gain
   (same finding the repo reached for A3B MoEs on the 3090). The custom 248320
   vocab also means you **can't borrow an external small Qwen3 draft** (those are
   151936-vocab).
2. **MTP self-speculation.** Community MTP grafts (`protoLabsAI`, `dan9070`,
   `wang-yang`, …) add a next-token head → `--spec-type draft-mtp` (GGUF) or the
   MLX-MTP builds, **no second model, free decode uplift** — the same mechanism
   that gave Qwen3.6-MTP ~1.24× on the M1 Max. Not in the official weights
   (`config.json` has no `num_nextn_predict_layers`), but **confirmed available
   for the 9B**: [`protoLabsAI/Ornith-1.0-9B-MTP-GGUF`](https://huggingface.co/protoLabsAI/Ornith-1.0-9B-MTP-GGUF)
   (KL-distilled head, **~1.4–1.7× on an A6000**, acceptance ~0.766,
   distribution-lossless; needs llama.cpp ≥ b9616). ✅ **The one real option** —
   and the fast fallback for the small cards (see
   [plan-rtx3060-12.md](plan-rtx3060-12.md#fallback-tier--if-the-offloaded-moes-run-too-slow--10-min-loop)).
3. **Gemma dense pair.** The 31B-Dense + a small Gemma-4 draft (the repo's proven
   dense spec-decode pattern) would work — but the 31B GGUF doesn't exist. ❌ moot.

## The self-improving angle — Ornith's native scaffold (future scope)

This is the most important conceptual point, and it's **deliberately out of scope
for the current comparison** — but worth documenting because it's where Ornith's
real upside lives.

**What Ornith is trained to do.** Unlike a plain coder, Ornith's RL optimizes the
*solution rollout **and** the scaffold that drives it* — the agentic loop, tool
calls, and search trajectory. The model wants to **generate and drive its own
agentic flow**, not just answer one prompt at a time.

**What this repo does.** The harness is the opposite by design: **Rust owns a
fixed `plan → execute → review` loop**, and every model call is a *stateless*
`opencode run` for one role. The scaffold is the harness's; the model only fills
in each step. (That's intentional — Rust owns all orchestration state; see
[orchestrator.md](orchestrator.md).)

So there are two ways to use Ornith here:

| Mode | What it measures | Status |
|------|------------------|--------|
| **Drop-in model in the fixed harness** | how well it codes *within our loop*, and whether it fits — **apples-to-apples** with every other model we've tested | ✅ **this comparison** |
| **Let Ornith drive its own scaffold** | the model's *self-generated* agentic flow — the capability it was actually trained for | 🔭 **future scope (postponed)** |

The current comparison uses **only mode 1** — Ornith as an ordinary coder slotted
into our plan/execute/review roles. It does **not** exploit scaffold generation,
so it establishes the **floor**, not the ceiling, of what Ornith can do.

**Why postpone, and what it would take.** Mode 2 needs a different harness path:
Rust would *supervise and sandbox* an Ornith-generated loop (capturing its
tool calls, enforcing the JSON/result contracts, providing the `cargo test`
oracle) rather than *imposing* the three roles. That's a meaningful design change,
not a config flag — hence future scope. But it is plausibly where **better code
quality** comes from, since co-optimizing scaffold + solution is the entire point
of Ornith's training. **Track it as the highest-value follow-up once the
drop-in numbers are in.**

## Per-system plan (4 boxes)

397B is out everywhere; 31B-Gemma is unavailable. So: **9B-Dense on the small
cards, 35B-MoE on the big two.**

| System | Model | Recommended quant | Fit | Speculative decoding |
|--------|-------|-------------------|-----|----------------------|
| **RTX 3060 12 GB** | 9B Dense | **Q6_K (7.4)** / Q8_0 (9.5) | whole, resident | ✅ **MTP self-spec** — [`protoLabsAI/Ornith-1.0-9B-MTP-GGUF`](https://huggingface.co/protoLabsAI/Ornith-1.0-9B-MTP-GGUF), ~1.4–1.7× (no external draft: custom 248320 vocab) |
| **RTX 5060 Ti 16 GB** | 9B Dense | **Q8_0 (9.5)** | whole, big headroom | solo (9B isn't MXFP4 → no FP4 lane). Stretch: 35B-Q4 + `--n-cpu-moe` |
| **RTX 3090 24 GB** | 35B MoE | **Q4_K_M (21.2)** whole | fits (tight, ~3B active → fast) | or **APEX-MTP (26.2)** w/ light `--n-cpu-moe` for self-spec |
| **M1 Max 64 GB** | 35B MoE | MLX **6-bit / 8-bit**, or GGUF Q6_K/Q8_0; full 262k ctx | easy | ⭐ **MLX-MTP** (`wang-yang/...MTPLX`) or GGUF APEX-MTP — self-spec |

> The harness gate requirements still apply: **native `tool_calls` llama.cpp can
> parse** + **strict-JSON envelopes**. Ornith uses a Qwen chat template (the cards
> note "Qwen chat template needs to be modified" for tool use) — confirm gate 1
> per runtime before trusting a loop run.

## Experiments (this branch)

Measured on the **M1 Max** (the box in hand; the NVIDIA cards are validated
separately). The headline is **MLX vs GGUF on Apple Silicon** for the 35B, plus
the 9B as the small-card stand-in:

1. **9B canary** — `bartowski` 9B **Q8_0** (GGUF/llama.cpp) and
   `mlx-community/Ornith-1.0-9B-8bit` (MLX/mlx-lm): confirm both runtimes load the
   `qwen3_5` arch + clear gate 1, and bench decode/prefill.
2. **35B MLX vs GGUF** — `mlx-community/Ornith-1.0-35B-6bit` vs `bartowski` 35B
   **Q6_K**, same prompt, decode + prefill. Quantifies the Apple-Silicon MLX speedup.
3. **MTP** (follow-up) — MLX-MTP / GGUF-APEX-MTP self-spec A/B vs the solo numbers.

## Measured results (M1 Max, 2026-06-28)

All on the M1 Max (64 GB). GGUF via `llama-server` (b9770), MLX via `mlx_lm.server`
(mlx-lm 0.31.3). Both runtimes load the `qwen3_5`/`qwen3_5_moe` arch and **clear
gate 1 — native `tool_calls`** (Ornith's Qwen template parses in llama.cpp *and*
mlx_lm.server).

**Throughput (`llama-bench` for GGUF; `mlx_lm` generate for MLX):**

| Model | Runtime / quant | Prefill | Decode | Peak mem |
|-------|-----------------|--------:|-------:|---------:|
| 9B Dense | GGUF Q8_0 (llama.cpp) | 542 (pp512) | **32.9** | ~9 GB |
| 9B Dense | MLX 8-bit | — | **~37** | ~9.7 GB |
| 35B MoE | GGUF Q6_K (llama.cpp) | 815 (pp512) | **40.7** | ~28 GB |
| 35B MoE | **MLX 6-bit** | 577 (1.7k-tok) | **55.6** | ~29.8 GB |

→ **MLX wins decode by ~1.35×** (35B: 55.6 vs 40.7; 9B: 37 vs 33) — Apple's
framework is better tuned for Metal. Prefill methodology differs (pp512 vs a
1.7k-token generate), so it's not directly comparable; decode is the robust signal.

**Wall-clock to working code (the headline metric) — Ornith-35B, M1 Max:**

| Runtime | Loop wall-clock | Steps | `cargo test` | Run shape |
|---------|----------------:|:-----:|:------------:|-----------|
| **MLX 6-bit** | **6m06s** | 3 (all *Continue*) | ✅ **3/3** | clean single crate (even added a `--times 0` test) |
| GGUF Q6_K | *(in progress)* | — | — | ⚠️ **wandered on first attempt** — the planner hallucinated a write to `/Users/brian/.../fish_greeting.fish` (auto-rejected as external dir), then retried |

The MLX run is clean and fast; the GGUF run exposed an Ornith reasoning-model
**wander** (off-task file write outside the workspace) under the llama.cpp Qwen
template — a tool/template-interaction quirk worth a closer look, not necessarily
a model-capability gap. Full cross-runtime + cross-hardware numbers consolidate
into [performance-analysis.md](performance-analysis.md) as the GGUF loop A/B and
the small-card (3060/5060) 9B runs complete.
