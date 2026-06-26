# Optimizations Plan — fast, low-power local coding (for the RTX 3090 node)

> ✅ **Implemented — see [nvidia-3090-poc-summary.md](nvidia-3090-poc-summary.md)
> for actual results.** This is the original *plan*; the build diverged from it.
> The plan assumed a Qwen2.5-Coder dense target + tiny draft via speculative
> decoding under one `qwen-coder` alias / `start-coder-specdecode.sh`. In practice
> Qwen2.5-Coder's tool-calling is broken in llama.cpp, and the winning config is a
> **Qwen3-Coder-30B-A3B MoE run solo**. Each model now has its **own alias + dir +
> start script** (`start-qwen3-coder.sh`, `start-gemma4-26b.sh`, `start-gemma4-31b.sh`,
> `start-qwable.sh`); the `qwen-coder`/`start-coder-specdecode.sh` names below are
> historical. Treat code/alias snippets here as illustrative, not current.

**Audience:** a fresh agent (a copy of Claude Code) running on the **RTX 3090
24 GB** workstation, tasked with implementing the next-generation serving config
for this harness. This doc is self-contained: it captures *why* (measured on the
3060 + dual-Xeon node) and *what to build* (on the 3090), so you can proceed
without re-deriving the analysis.

> Read alongside: [performance-analysis.md](performance-analysis.md),
> [arch-nvidia-3060-poc-summary.md](arch-nvidia-3060-poc-summary.md),
> [glm-models.md](glm-models.md), and [config/](config/README.md). The harness
> design is in [orchestrator.md](orchestrator.md).

---

## TL;DR — what to build on the 3090

Serve a **non-reasoning 32B coder target + a tiny same-family draft, speculative
decoding, fully VRAM-resident**:

- **Target:** `Qwen2.5-Coder-32B-Instruct` @ **Q4_K_M** (~19–20 GB)
- **Draft:** `Qwen2.5-Coder-1.5B-Instruct` @ Q4 (~1 GB) via `--model-draft`
- Both on the 3090 (24 GB), `-ngl 99`, flash-attention on, q8_0 KV.
- Expected: **~50–90 tok/s decode**, near-frontier *coding* quality, minutes/task,
  a fraction of GLM-5.2's energy.
- Keep a **reasoning profile** (`R1-Distill-Qwen-32B` + `R1-Distill-Qwen-1.5B`
  draft) for the occasional genuinely-hard step.

The harness needs **no code changes** for this (a fast GPU-resident model behaves
like Qwable, which already works end-to-end here). Just build llama.cpp with
CUDA, download the GGUFs, drop in a start script + an `opencode.json` provider,
and run `scripts/orchestrate-stepwise.sh` or `scripts/demo-orchestrate.sh`.

---

## Background — what the 3060 + dual-Xeon node measured (the "why")

Hardware there: dual Xeon E5-2697 v4 (Broadwell, AVX2, **no AMX/tensor cores on
CPU**), DDR4-2400 quad-ch ×2 (~150 GB/s aggregate), **503 GB RAM**, **RTX 3060
12 GB**, llama.cpp CUDA build b9784.

| Model | Type | Decode | Prefill | Verdict |
|---|---|---:|---:|---|
| **Qwable-v1 IQ4_XS** (qwen35moe, 35B-A3B, 18.9 GB) | MoE, experts→RAM | ~44–49 t/s | ~413 t/s | works great, GPU-friendly |
| **GLM-5.2 UD-IQ3_XXS** (754B-A40B, 282 GB) | MoE, experts→RAM | **~1.4 t/s** | ~12 t/s | **too slow + power-hungry**; PLAN call failed after 53 min (see below) |

### The governing principle (decides everything below)
Single-stream inference has two regimes:
- **Working set in VRAM** → GPU-resident → **fast (40–80 t/s) and energy-cheap**
  (GPU ~300 W for *minutes*).
- **Working set in system RAM** (model too big for VRAM) → **RAM-bandwidth-bound
  and energy-expensive** (whole dual-Xeon + RAM at ~290 W+ for *hours*).

GLM-5.2 decode ≈ 1.4 t/s because each token drags ~16 GB across DDR4 (NUMA-
penalized). **The win on both axes you care about — speed and electricity — is to
keep the per-token working set in VRAM.** That is the entire rationale for the
3090 plan.

### Two failures worth remembering
1. **GLM-5.2 PLAN failed after ~53 min** not from hardware but from the opencode
   `build` agent's **~7,500-token system prompt** forcing ~10-min prefills per
   *turn* at GLM's speed, then never emitting the JSON. Fast GPU-resident models
   don't hit this (qwable's agent turns are seconds). So on the 3090 the heavy
   agent is fine; **no need for the "direct-completion" harness fix** (see
   "Harness notes"). That fix is only worth doing if you ever go back to slow
   RAM-resident models for overnight batch.
2. Bigger-RAM / 2×P40 hardware does **not** rescue GLM-5.2: decode is RAM-
   bandwidth-bound and a P40 (Pascal, no tensor cores, crippled FP16, weak FA) is
   a *worse* compute engine than the Ampere 3090. More VRAM only lets you *fit*
   more; the 3090 lets you go *fast*. **Prefer the 3090.**

---

## Key technical facts (don't re-derive these)

1. **Speculative decoding is lossless w.r.t. the target.** The verify step
   accepts/resamples so the output distribution is *identical* to sampling the
   target alone → **you always get the target's quality, regardless of the
   draft.** The draft only affects *speed*.
2. **You get the target's speed × ~1.5–2.5×, not the draft's raw speed.** The
   target verifies every token; the draft just lets it verify several per forward
   pass. Coding tasks draft well (predictable) → high end of the range.
3. **Drafts must be genuinely small (0.5–1.5B), not just low-bit.** A Q4 copy of
   the same 32B is ~as expensive as the target → no gain. Use a tiny *sibling*.
4. **Draft and target MUST share the same tokenizer/vocab** (same model family).
   - ✅ Qwen draft + Qwen-family target (incl. R1-Distill-Qwen, Qwen-Coder)
   - ✅ Llama draft + Llama-family target (incl. R1-Distill-Llama)
   - ❌ cross-family (Llama↔Qwen↔GLM). This is why "Qwable→GLM" spec-decoding is
     impossible.
5. **Acceptance ∝ style match.** For a *reasoning* target, the matching small
   *reasoning-distill* draft (e.g. R1-Distill-Qwen-1.5B) accepts better than a
   Coder draft on the thinking tokens. For a *coder* target, use a Coder draft.
6. **Non-reasoning models often win for agentic coding.** The metric is
   **wall-clock (and energy) per task = tokens_generated ÷ tok/s**. Reasoning
   models inflate the numerator 3–10×; most agentic steps are mechanical. Default
   to a non-reasoning coder; reserve reasoning for hard steps. (Qwen3 can toggle
   per-call: `enable_thinking:false` / `/no_think`.)
7. **FP16 of a 32B doesn't fit 24 GB** (~64 GB). Best target quant on a 3090 is
   **Q4_K_M / Q5_K_M**; quant the draft light (Q4) — its quality barely matters,
   only its agreement rate.

---

## Primary implementation — 3090 spec-decode coder

### 0. Prereqs on the 3090 box
- NVIDIA driver + CUDA toolkit (`nvcc`), cmake, ninja, gcc, git.
- The 3090 is **sm_86** (same as the 3060) → the existing build flags apply.

### 1. Build llama.cpp with CUDA (fresh, recent)
```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
# If nvcc rejects a too-new gcc (CUDA vs gcc mismatch), add:
#   -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler"
cmake --build build --config Release -j
build/bin/llama-server --list-devices    # confirm the 3090 is seen
build/bin/llama-server --help | grep -i draft   # confirm spec-decode flags
```

### 2. Download GGUFs (coder profile)
Place on the roomiest mount. Pick reputable GGUF repos (e.g. Qwen official or
bartowski). Verify exact filenames/sizes from the repo before downloading.
```
TARGET: Qwen2.5-Coder-32B-Instruct  Q4_K_M   (~19–20 GB)
DRAFT : Qwen2.5-Coder-1.5B-Instruct Q4_K_M   (~1 GB)   # or 0.5B for an even cheaper draft
```
(If `huggingface-cli`/`hf` is unavailable, reuse the curl+wget pattern in
`docs/config/arch-nvidia-3060/download-glm.sh` — point it at the coder repo.)

### 3. Start script — `docs/config/nvidia-3090/start-coder-specdecode.sh`
Create it (mirror the existing `arch-nvidia-3060/start-qwable.sh` structure):
```bash
exec llama-server \
  -m  /path/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf \
  --model-draft /path/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  --alias qwen-coder \
  --gpu-layers 99 --gpu-layers-draft 99 \
  --draft-max 16 --draft-min 4 --draft-p-min 0.6 \
  --ctx-size 32768 --parallel 1 --cont-batching \
  --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --host 127.0.0.1 --port 8080 --n-predict 4096 --api-key local
```
Notes:
- This is a **non-reasoning** model → **no `--reasoning-budget` needed**.
- Tune `--draft-max` (8–16) and `--draft-p-min` (0.4–0.8) and re-bench; higher
  draft-max helps when acceptance is high (code), hurts when it's low.
- Watch VRAM: target Q4_K_M (~20 GB) + draft (~1 GB) + KV should fit 24 GB at
  32k ctx; if tight, lower `--ctx-size` or use a 0.5B draft.
- If you want simultaneous models on one box, give each its own port (8080 may be
  taken by an opencode UI — it was on the 3060 box; check `ss -ltnp | grep 8080`).

### 4. opencode provider — `~/.config/opencode/opencode.json`
```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llamacpp": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://localhost:8080/v1", "apiKey": "local" },
      "models": {
        "qwen-coder": {                 // == llama-server --alias
          "name": "Qwen2.5-Coder-32B (spec-decode)",
          "tools": true,
          "cost": { "input": 0, "output": 0 },   // REQUIRED: opencode 1.x cost-calc throws DecimalError without it
          "limit": { "context": 32768, "output": 4096 }
        }
      }
    }
  },
  "model": "llamacpp/qwen-coder"
}
```
Verify: `opencode models | grep llamacpp` then a smoke test:
`opencode run 'reply with OK' --model llamacpp/qwen-coder`.

### 5. Run the harness
```bash
cargo build --release
# quick path test:
PLAN_ONLY=1 MODEL=llamacpp/qwen-coder ./scripts/demo-orchestrate.sh
# full loop (fast model → run it straight through):
MODEL=llamacpp/qwen-coder ./scripts/demo-orchestrate.sh
# or chunked / resumable:
MODEL=llamacpp/qwen-coder WS=/path/run1 ./scripts/orchestrate-stepwise.sh plan
MODEL=llamacpp/qwen-coder WS=/path/run1 ./scripts/orchestrate-stepwise.sh step   # repeat
```

### 6. Benchmark & validate (record results)
```bash
# base vs spec-decode decode rate:
llama-bench -m TARGET.gguf -ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 128
# (spec-decode throughput is best measured via the server's per-request timings)
```
Capture: decode t/s (base vs spec), prefill t/s, VRAM used, and — most important —
**wall-clock for the full greeter loop** and **tokens generated** (the real metric).
Then write a POC summary mirroring `arch-nvidia-3060-poc-summary.md`.

---

## Ranked alternatives / fallbacks

1. **(chosen) Qwen2.5-Coder-32B + 1.5B draft, spec-decode, 3090.** Best speed +
   power per coding quality. Non-reasoning → fewest tokens/task.
2. **R1-Distill-Qwen-32B + R1-Distill-Qwen-1.5B draft (reasoning profile).** Same
   hardware fit; use for hard algorithmic/debug steps where reasoning pays. More
   tokens/task → slower wall-clock; keep as a second provider, not the default.
3. **Qwen3-32B (thinking toggizable) + Qwen3-1.7B draft.** Flexible: thinking off
   for routine steps, on for hard ones, single model.
4. **No-draft single model** (e.g. Qwen2.5-Coder-14B Q4) if spec-decode setup is
   fiddly — still GPU-resident and fast; lower ceiling.
5. **DeepSeek-R1 1.58-bit (671B-A37B, ~131 GB) in RAM** — only if you have a
   big-RAM box and want max quality overnight; ~2× GLM-5.2 decode but still
   RAM-bound/batch-only/high-energy. Not for the 3090.
6. **GLM-5.2-in-RAM** — overnight max-quality batch only; needs the direct-
   completion harness fix to be usable. Avoid for interactive/power-sensitive.

---

## Harness notes (state of the code you'll pull)

- **File-based JSON hand-off (already in):** each role tells opencode to write its
  JSON to `.bootstrap/*.llm.json`; the harness reads that file (stdout fallback).
  Works because opencode 1.x is agentic and prefers tool calls over inline prose.
  See `src/prompts.rs::append_write_json` + `src/orchestrate.rs::capture_json`.
- **Resumable / stepwise (already in):** `--resume` continues from
  `.bootstrap/state.json`; `--max-steps N` caps steps per invocation. Driver:
  `scripts/orchestrate-stepwise.sh` (`plan` / `step` / `status` / `reset`). Built
  to spread slow runs over chunks — less critical for a fast 3090 model, but handy.
- **opencode gotchas (learned the hard way):**
  - opencode **1.x** required (0.2.x throws `DecimalError` vs the new llama.cpp).
  - Every model needs a **zero `cost` block** or opencode 1.x's cost-calc throws
    `DecimalError`.
  - The default **`build` agent** is the only reliable one here; custom/`plan`
    agents hung. Don't route roles through custom agents.
  - **Reasoning-distilled models** need `--reasoning-budget 0` (server) or they
    spend the whole `--n-predict` budget thinking before answering. Non-reasoning
    coders don't need this.
- **Direct-completion fix (NOT implemented; optional):** for slow RAM-resident
  models, route the no-tools planner/reviewer to llama-server's
  `/v1/chat/completions` directly (bypass the heavy `build` agent → drops their
  prompt from ~7,500 to ~600 tokens). **Skip this on the 3090** — fast models make
  the agent overhead negligible. Implement only if you revive GLM-5.2/R1-in-RAM.

---

## M1 Max 64 GB variant (if used)

No CPU tensor cores → spec-decode helps *less* (verify runs on shader ALUs), so
the **fewer-tokens lever dominates**: prefer a **non-reasoning coder**.
- `Qwen2.5-Coder-32B` Q4 (~19 GB) ≈ 15–18 t/s, or `Qwen2.5-Coder-14B` Q4 for
  snappier interactive (~25–30 t/s). Add a 0.5–1.5B same-family draft for a modest
  bump.
- Unified 64 GB can hold 70B-Q4 (~40 GB) but ~8–10 t/s → batch only.
- Build llama.cpp with Metal (default on Apple Silicon). Same opencode.json,
  same harness, same `cost`/port rules.

---

## Success criteria

- [ ] llama.cpp CUDA build runs on the 3090; spec-decode flags present.
- [ ] Server serves target+draft, VRAM < 24 GB at 32k ctx, smoke test passes.
- [ ] `opencode models` lists the model; a `run` returns clean text.
- [ ] Full greeter loop completes via the harness with a working, tested CLI.
- [ ] Recorded: decode t/s (base vs spec), prefill t/s, VRAM, **wall-clock/loop**,
      **tokens/loop**, and rough watts × time (energy/task) — compare to the 3060
      qwable and GLM-5.2 rows in `performance-analysis.md`.
- [ ] (Optional) reasoning profile (R1-distill) wired as a second provider.
- [ ] Write `docs/nvidia-3090-poc-summary.md` mirroring the 3060 POC format.

---

## Open questions to resolve on the 3090 box

- Exact best `--draft-max` / `--draft-p-min` for this draft+target on coding
  (bench a few; code usually likes draft-max 12–16).
- Whether Q5_K_M target fits comfortably with the draft + 32k KV in 24 GB (try;
  fall back to Q4_K_M).
- Confirm the chosen GGUF repos' tokenizers match between target and draft
  (they must; same family by construction, but verify the draft loads as a draft
  without a vocab-mismatch warning).
- Is opencode's `build` agent's per-turn prefill acceptable at this model's
  prefill rate? (Should be — 3090 prefill is fast — but measure.)
