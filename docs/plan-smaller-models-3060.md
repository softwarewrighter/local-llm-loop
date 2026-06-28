# Plan — testing *even smaller* models on the RTX 3060 12 GB

Follow-up to [nvidia-3060-12-results.md](nvidia-3060-12-results.md) (where the
smallest model that produced correct code was **Ornith-9B**). Goal: push **below
9B** — 2B–4B native-MTP models and draft+target pairs — and find what, if
anything, still **codes correctly** while running **faster** than Ornith-9B-MTP's
4m29s. Seeded by a research thread ([small-llm-research.txt](small-llm-research.txt)).

## Two distinct bets (don't conflate them)

| Bet | Question | Our prior (measured) |
|-----|----------|----------------------|
| **A. Smaller standalone coder** | Does a 2B–4B model clear both gates **and** emit a building, `cargo test`-green crate? | ⚠️ **Pessimistic.** Every *general* model ≤14B we ran failed the code floor (Qwen3-8B completed the loop but the crate didn't build); only the *purpose-trained* Ornith-9B passed below 14B. A 4B general model is **below the floor** on prior evidence. |
| **B. Faster via spec-decode** | Can a small **draft** (or a native-MTP small model) speed up decode while a real coder verifies? | ✅ **Promising for dense.** MTP gave Ornith-9B ~1.3–1.7×. But ⚠️ spec-decode is **break-even-to-negative on A3B MoEs** on Ampere (our finding + thc1006), so drafts only help **dense** targets. |

**The honest expectation:** sub-9B general models most likely become **good draft
models**, not standalone coders. The headline experiment is whether *any* 2B–4B
model crosses the code floor — if one does, it would beat 4m29s.

## What's actually available (verified on HF, 2026-06-28)

| Model | GGUF | MTP? | Vocab / draft | Notes |
|-------|------|:----:|---------------|-------|
| **Gemma-4-E4B-it** | `unsloth/gemma-4-E4B-it-GGUF` (Q4_K_M + **bundled `MTP/…-MTP.gguf`**) | ✅ bundled + standalone head | Gemma family | effective ~4B; native MTP. **Top small-MTP candidate.** |
| **Gemma-4-E2B-it** | `unsloth/gemma-4-E2B-it-GGUF` (Q4_K_M + **MTP**) | ✅ | Gemma | effective ~2B; smallest native-MTP test. |
| **Qwen3-4B** | `unsloth/Qwen3-4B-GGUF` (Q4_K_M/Q5_K_M) | ✗ | 151936 — pairs w/ Qwen3-0.6B | dense; classic draft+target target. |
| **Qwen3-0.6B** | `unsloth/Qwen3-0.6B-GGUF` | ✗ (draft) | 151936 (exact) | the draft for Qwen3-4B/8B. |
| Qwen2.5-Coder-3B | `unsloth/Qwen2.5-Coder-3B-Instruct-GGUF` | ✗ | — | ⚠️ **gate-1 fail** (tool-calls broken in llama.cpp) → speed-only. |
| ~~Qwen3.6-4B~~ | — | — | — | **not released** as a separate GGUF (the research doc assumed it). Use Qwen3-4B. |

> **Caveat carried from our 35B run:** an "MTP" GGUF doesn't always contain the
> head — the unsloth Qwen3.6-35B GGUF **lacked** MTP layers. **Verify the head
> loads** (`--spec-type draft-mtp` fails fast with "model doesn't contain MTP
> layers") before trusting the self-spec path. Also unknown: whether llama.cpp's
> MTP support covers **Gemma-4** (the research doc says it's "primarily Qwen
> 3.5/3.6") — this is the first thing to confirm.

## Experiment list (ranked; each ties to a measurable question)

1. **Gemma-4-E4B-it + MTP** — *does a ~4B native-MTP model code, and does Gemma
   MTP even load in llama.cpp?* Verify head loads → gate 1 → loop + `cargo test` →
   MTP A/B (`SPEC_OFF`) for decode uplift + acceptance. **Highest value:** answers
   both the Gemma-MTP support question and bet A at 4B.
2. **Gemma-4-E2B-it + MTP** — *how low can the floor go?* Same protocol at ~2B.
3. **Qwen3-4B + Qwen3-0.6B draft** (`--spec-type draft-simple`, exact vocab) —
   *classic spec-decode on dense small Qwen*: gate 1 → loop → decode + acceptance.
   Tests bet A for Qwen-4B and gives the draft-vs-MTP comparison the research doc
   wants (same family, clean A/B).
4. **Qwen2.5-Coder-3B + 0.5B** — **speed-only** reference (expected gate-1 fail);
   run `llama-bench` + acceptance to characterize a strong coder draft pair, even
   though it can't drive the loop.

All resident (no offload needed at these sizes); huge KV headroom on 12 GB. Reuse
the harness exactly as in the measured run (`demo-orchestrate.sh` + independent
`cargo test`); record VRAM, decode/prefill, acceptance, loop wall-clock, steps,
retries, build/test pass.

## Pass/fail bar

Same as the measured set: **clears gate 1 + `cargo test` green**. A model that
completes the loop but produces a non-building crate is logged as **draft-grade,
not coder-grade** (the Qwen3-8B outcome) — still useful for bet B.

## Phase 2 — fine-tuning small models to code Rust (the likely next phase)

If the drop-in 2B–4B models confirm the prior (complete the loop but emit
non-building crates), the path forward is **not more prompting — it's training.**
Our own measured ICL finding says so: in-context techniques (few-shot, `emit`
self-heal, the ICRL retry loop) **scale a model's protocol-following, not its
coding ability** ([nvidia-5060-poc-summary.md](nvidia-5060-poc-summary.md):
qwen3-8b *completes* the loop after 12 ICRL rescues yet still can't write a valid
crate). So context-engineering closes the **gate** gap; only weight-level training
closes the **code-correctness** gap. Two distinct training targets:

| | **Bet A — small coder** | **Bet B — acceptance-draft** |
|--|--------------------------|------------------------------|
| Goal | a 2B–4B that *itself* writes building, tested Rust + clean envelopes | a 0.5–2B *draft* that maximizes the verifier's accept rate |
| Objective | code correctness + tool/JSON protocol | agreement with the target (not cross-entropy) |
| Difficulty | hard (must actually code) | easier (well-defined, narrow) |
| Deploy | standalone on 3060/5060 | speculative draft for a dense target |
| Vocab | free choice | **must match the target's tokenizer** (Qwen 151936; can't draft Ornith's 248320 or cross-family) |

**The asset we already have: a verifier.** The harness's `cargo test` oracle turns
this into a **rejection-fine-tuning / STaR loop** ([Zelikman 2022](https://arxiv.org/abs/2203.14465))
— no human labels needed:

1. **Generate** (teacher on the RTX 3090/24 GB): run Ornith-35B / Qwen3-Coder-30B
   through *our own loop* over many Rust specs → spec → plan → tool calls → code →
   `cargo test`. **Keep only trajectories whose crate builds + passes.**
2. **Train** (student on the 3090): LoRA/QLoRA a small base (Qwen3-1.7B/4B for
   vocab-compatibility with the draft path, or SmolLM2 / Gemma-E2B) on the verified
   trajectories. **unsloth** is the fit — QLoRA of a 4B fits 24 GB easily, fast,
   and **exports GGUF directly** into our llama.cpp pipeline.
3. **Eval** (student on the 3060): run it through *this exact harness* (loop +
   independent `cargo test`) — the eval already exists; the greeter spec + a
   held-out spec set are the rubric. Measure gate pass, build/test pass, wall-clock.
4. **Iterate**: feed the student's *own* passing trajectories back into the
   training set (self-improvement); later, RL with `cargo test` as the reward
   (the Ornith recipe, scaled down).

**What to target first** (from our failure data): (a) **valid building Rust**
(edition 2024, valid Cargo.toml, single-crate layout — the exact things Qwen3-8B
botched), and (b) **strict-JSON envelopes + native tool-calls** (gates 1–2). The
synthetic dataset should be `cargo fmt` + `clippy`-clean and carry the harness's
envelope schema as the output format.

**Scope reality:** this is a **dataset-generation + training project**, not a
config flag — a meaningful new effort on the 3090. But it's the only lever that
moves capability, and we already own the two hardest pieces (a teacher and an
automated correctness oracle). Track Bet B (the acceptance-draft) as the cheaper
first cut and Bet A (the standalone coder) as the ambitious goal. The "Mike Coder"
idea in [small-llm-research.txt](small-llm-research.txt) — bias the dataset toward
*our* conventions (architecture, macros, DSL, TDD) — layers on top of either.

## Measured results (this box, 2026-06-28)

### Experiment 1 — Gemma-4-E4B-it (~4B effective, 7.52B total) ✅ **codes**

| Metric | Value |
|--------|-------|
| Gate 1 (native tool_calls) | ✅ pass |
| Loop wall-clock | **9m53s** |
| Steps / retries | 9 / **30** |
| `cargo test` (independently re-run) | ✅ **2/2** — real `pub fn greet()` lib + 2 unit tests, edition 2024, clap-derive |
| Decode `tg128` (llama-bench) | **78.9 t/s** (fa on) / 79.5 (fa off) |
| Prefill `pp512` | ~2,800 t/s |
| VRAM (resident) | ~3.5 GB |

**This breaks the "≤14B can't code" floor — barely.** Gemma-4-E4B is the smallest
model in our testing to emit a building, tested crate. But the cost is **30 ICRL
retries** (vs **0** for Ornith-9B): it *can* code, only with heavy harness rescue.
And note the paradox — it **decodes ~2× faster than Ornith-9B (79 vs 41 t/s)** yet
its loop is **slower** (9m53s vs 4m29s) **because the retries dominate, not decode.**
→ For small models the bottleneck is **protocol/coding reliability, not speed** —
which is exactly the argument for the Phase-2 fine-tuning above (and confirms the
ICL finding: prompting rescues the gate, not the capability).

### Gemma-4 MTP — works, but flash-attn must be OFF

- `--model-draft mtp-gemma-4-E4B-it.gguf --spec-type draft-mtp` **engages** —
  measured **draft acceptance 0.80** (164/204, mean accepted len 2.61).
- ⚠️ **Config trap:** with `--flash-attn on` the draft context creation **crashes**
  (CUDA fatal in `ggml-cuda/fattn.cu:110`). Must run **`--flash-attn off`** (fa
  barely affects decode here, ~79 t/s either way). Wired into the start script
  (`FLASH_ATTN` defaults to `off` when MTP is on). Also drop q8_0 KV with fa off.
- MTP won't move this model's loop wall-clock much — the loop is retry-bound, not
  decode-bound — so MTP here matters more as a *draft-model* validation than a
  coder speedup.

### Experiment 2 — Gemma-4-E2B-it (~2B effective, 3.1 GB) ❌ **does not code**

| Metric | Value |
|--------|-------|
| Gate 1 | ✅ pass |
| Loop wall-clock | 4m18s |
| Steps / retries | 8 / **0** |
| `cargo test` | ❌ **stub only** — produced the bare `cargo new` boilerplate (`fn main(){ println!("Hello, world!"); }`), no `greet()`, no tests, no clap; "builds" trivially, 0 tests |

**The floor sits between 2B and 4B.** E2B confidently declared done on boilerplate
(**0 retries**) — the opposite failure mode from a struggling-but-correct model.
Telling contrast: **E4B needed 30 retries to get it right; E2B needed 0 to get it
wrong.** Retry count alone doesn't signal capability — a model below the floor
doesn't even know it's failing. (Also scattered a stray `src/main.rs` into the
workspace root — the file-placement wandering small models show.)

### Experiment 3 — Qwen3-4B (dense general) + Qwen3-0.6B draft ❌ **does not code**

| Metric | Value |
|--------|-------|
| Gate 1 | ✅ pass |
| Loop wall-clock | 15m32s (reasoning model — verbose `<think>`) |
| Steps / retries | 6 / 3 |
| `cargo test` | ❌ **doesn't build** — malformed `Cargo.toml` (`[crate-type="bin"]` → "invalid table header"), files scattered at workspace root (`main.rs`, `greet_test.rs`), no lib |
| Classic draft acceptance (`draft-simple`) | **0.41–0.55** (mean len ~3) |

Confirms general small models fail the floor regardless of family (matches the
prior Qwen3-8B result). **And the spec-decode finding:** classic external-draft
acceptance (~0.5) is **well below MTP self-spec** (Gemma 0.80, Ornith 0.73–0.85) —
so even for *bet B*, a bundled MTP head beats a separate draft model here.

### Code-floor summary (this harness, hardened)

| Model | Size | Codes? | Retries | Note |
|-------|-----:|:------:|:-------:|------|
| Ornith-1.0-9B | 9B (purpose-trained) | ✅ | 0 | clean |
| **Gemma-4-E4B** | ~4B | ✅ | 30 | **the floor** — barely, heavy rescue |
| **Gemma-4-E2B** | ~2B | ❌ | 0 | stub only (confidently wrong) |
| **Qwen3-4B** | 4B general | ❌ | 3 | malformed Cargo.toml, scattered files |
| Qwen3-8B (prior, 5060) | 8B general | ❌ | — | non-building crate |

→ The floor is **model-specific, not size-monotonic**: a 4B *Gemma* codes where
4B and 8B *general Qwen* don't — **architecture/training matters more than
parameter count.** Gemma-4-E4B is the only sub-9B general model that crosses it,
and only with heavy harness rescue. This is the empirical case for **Phase 2**:
to get reliable sub-9B Rust, fine-tune — don't just prompt.

### Experiment status

| # | Experiment | Status |
|---|-----------|--------|
| 1 | Gemma-4-E4B-it (+MTP) | ✅ **codes** — 2/2, 9m53s/30 retries; MTP works fa-off, accept 0.80 |
| 2 | Gemma-4-E2B-it (+MTP) | ❌ **stub only** — below the floor; 4m18s/0 retries |
| 3 | Qwen3-4B + Qwen3-0.6B draft | ❌ **non-building** — 15m32s/3 retries; classic draft accept ~0.5 |
| 4 | Qwen2.5-Coder-3B + 0.5B | ⏸️ skipped — speed-only (gate-1 fail expected); low marginal value |

## Conclusions

1. **The sub-9B code floor is real and sits at ~4B — but only for the right
   model.** Gemma-4-E4B is the smallest model that produces a building, tested
   crate; E2B and general Qwen-4B/8B do not. **Architecture/training > size.**
2. **Crossing the floor ≠ being practical.** E4B needs 30 ICRL retries (9m53s);
   Ornith-9B-MTP does it in 0 retries (4m29s). For *drop-in* use, **Ornith-9B-MTP
   remains the 3060 pick** — going smaller costs more reliability than it saves.
3. **MTP self-spec > classic external draft** for acceptance (0.80 vs ~0.5) — and
   Gemma-4 MTP needs **flash-attn off** in this build.
4. **The path to reliable sub-9B Rust is training, not prompting** (Phase 2). The
   floor models *can't be prompted into competence* — E4B only reaches it via
   brute-force retries, and E2B/Qwen-4B never do.

## Next action

Drop-in exploration is **complete**. The decision point: pursue **Phase 2
fine-tuning** (stand up the 3090 data-gen + LoRA pipeline — the real capability
lever), keep **Ornith-9B-MTP** as the production 3060 model, and optionally use
**Gemma-4-E4B** as a *fast draft* candidate for a future Gemma-family target.
