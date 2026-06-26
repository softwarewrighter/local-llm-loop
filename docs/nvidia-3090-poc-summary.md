# Proof-of-Concept Summary — Arch Linux / NVIDIA RTX 3090 24 GB (fast GPU-resident coder)

The [optimizations plan](optimizations-plan.md) asked for a **fast, low-power,
fully-GPU-resident** coding model on the RTX 3090 to replace the slow RAM-resident
GLM-5.2 path, and to run the planner → executor → reviewer loop
(`bootstrap orchestrate`) end-to-end. This is the result. The model search did
**not** land where the plan guessed (a Qwen2.5-Coder dense target + tiny draft via
speculative decoding); it landed on a **Qwen3-Coder MoE run solo**, for concrete,
measured reasons documented below.

- **Date:** 2026-06-25
- **Winning model (all three roles):** `llamacpp/qwen-coder` =
  **Qwen3-Coder-30B-A3B-Instruct Q4_K_M** (MoE, 30.5B total / **~3.3B active**),
  fully GPU-resident, **no draft** (speculative decoding is moot on A3B — see below)
- **Host:** Arch Linux · **RTX 3090 24 GB** · llama.cpp CUDA build **b9728**
  (`/usr/bin/llama-server`) · **opencode 1.17.11**
- **Server:** `-ngl 99 --ctx-size 32768 --parallel 1 --flash-attn on
  --cache-type-k q8_0 --cache-type-v q8_0 --jinja` (see
  [config/nvidia-3090/start-qwen3-coder.sh](config/nvidia-3090/start-qwen3-coder.sh))
- **VRAM:** **~20.8 GB / 24 GB** (whole model + 32k KV in VRAM)
- **Throughput:** decode **~156 tok/s**, prefill **~1540 tok/s** (single-stream)
- **Command:** `MODEL=llamacpp/qwen-coder ./scripts/demo-orchestrate.sh`
- **Outcome:** **working, independently-verified Rust CLI**; loop ran all 6 steps
  to a natural completion (6× Continue, no fail-safe Stop). Wall-clock ~210 s.

## Result — the loop completed and the CLI works

The planner produced a 6-step plan; the executor implemented each step (real tool
calls, real files), and the reviewer returned **Continue** six times to a natural
end. Independent verification (run by a human after the loop halted):

- `cargo test` → **3 passed** (`test_greet_times_0/1/3`)
- `greet Alice` → `Alice` (exit 0)
- `greet Bob --times 3` → three lines (exit 0)
- `greet X --times 0` → `Error: Times must be greater than 0` (exit **1**)

Met the spec: pure `greet(name, times) -> String`, clap-derive parsing, I/O
isolated in `main()`, unit tests for times 1/3 (plus a times-0 test). **Minor
deviation:** the greeting is the bare repeated name (`Alice`) rather than
`Hello, Alice!` — a loose-but-valid reading of "a friendly greeting" (the
macOS/3060 runs produced the `Hello, X!` form).

## The model search — what was tried and why most failed

Getting here required ruling out the plan's assumed config. Every pairing was run
on the same 3090, llama.cpp b9728, `--spec-type draft-simple`, `--parallel 1`.
Two independent failure modes decided everything: **(A) does llama.cpp parse the
model's tool calls into native `tool_calls`?** (opencode is agentic — if not, the
executor "narrates" work it never does), and **(B) does the model emit strict,
parseable JSON?** (the harness uses `serde_json`).

> **Three distinct Qwen-lineage models appear below — do not conflate them:**
> **Qwen2.5-Coder** (the *2.5* dense coder, failed on tool-calling),
> **Qwen3-14B** (*base* Qwen3 — **not** a coder — paired for spec-decode, failed on
> JSON), and **Qwen3-Coder-30B-A3B** (the *3.x* MoE **coder**, which succeeded).
> They are three different models; only the last is the working coder.

| Target + draft | Spec-decode | (A) Tool calls | (B) Strict JSON | Loop |
|---|---|---|---|---|
| Qwen2.5-Coder-32B Q4 + 1.5B | — | — | — | **OOM** at 32k (21.7 GB weights) |
| Qwen2.5-Coder-14B **Q4** + 1.5B Q4 | ✅ 133 t/s, **84% accept** | ❌ wrong delimiter (`<tools>`) | — | executor narrates, no files |
| Qwen2.5-Coder-14B **Q5** + 1.5B Q8 | ✅ engaged | ❌ **same** wrong delimiter | — | (not a quant artifact) |
| Qwen3-14B Q4 + 0.6B Q8 | ✅ 97 t/s, 55% accept | ✅ native `tool_calls` | ❌ trailing comma / bad escaping | plan parse fails |
| **Qwen3-Coder-30B-A3B Q4 (solo)** | n/a (MoE) | ✅ **native** | ✅ **clean** | ✅ **completes, verified** |
| **Gemma-4-26B-A4B Q4 (solo)** | n/a (MoE) | ✅ **native** | ✅ **clean** | ✅ **completes, verified — best code** |
| **Gemma-4-31B Q4 + E2B Q4 draft** | ✅ 36→46 t/s, 44% accept | ✅ native | ✅ clean | ✅ **completes, verified** (slow; flaky path 1/2) |
| gpt-oss-20b MXFP4 (solo) | n/a (MoE) | ✅ native | ❌ literal control char in JSON | plan parse fails |
| Granite-4.1-8B Q4 (solo) | n/a | ✅ native | ❌ invalid JSON + wrong path | plan parse fails |

### Verified-working set (besides Qwable)

Four models (incl. Qwable) complete the loop and produce independently-verified
code on this box — across three vendors, one a dense spec-decode pair:

| Model | Arch | Decode | Greeter result |
|---|---|---|---|
| **Gemma-4-26B-A4B** | MoE solo | ~142 t/s | 2/2 tests; `Hello, X!` (best quality) |
| **Qwen3-Coder-30B-A3B** | MoE solo | ~189 t/s | 3/3 tests; bare name |
| **Gemma-4-31B + E2B** | dense + spec-decode | 36→~46 t/s | 2/2 tests; `Hello, X!`; richest review (Continue/Insert/Skip) |
| Qwable-v1 (baseline) | MoE solo | ~158 t/s | (prior POCs) |

**Gemma-4-31B + E2B** is the working **dense speculative-decoding coding pair**
(exact 262144 vocab match; `--spec-type draft-simple`; ~1.3× over solo). It runs at
ctx 16384 to fit the draft in 24 GB. Two caveats: it's **slow** (dense → ~12-min
loop) and showed a **flaky path bug** (1 of 2 runs wrote an absolute `/.bootstrap/`
path that opencode auto-rejected; the retry succeeded).

Three candidates clear tool-calling but **fail the clean-JSON gate**, so they don't
complete the loop today:
- **gpt-oss-20b** — literal control char in its JSON envelope; *fastest measured*
  (~205 t/s decode).
- **Granite-4.1-8B** — invalid JSON *and* mangled the `.bootstrap/` path (dropped
  the dot); 8B is too small for reliable strict JSON.
- **Qwen3-14B** pair — trailing comma / unescaped quotes.

They become viable only if model output is constrained to valid JSON at the source
(grammar / json-schema) — see next steps. **Pattern:** the reliable models are the
larger MoE/dense coders (≥26B); sub-15B models flunk strict JSON.

Conclusions from the matrix:

1. **Qwen2.5-Coder has broken tool-calling here, at any quant.** Both Q4_K_M and
   Q5_K_M emit their call wrapped in `<tools>…</tools>` (the *input* delimiter)
   instead of `<tool_call>…</tool_call>`, so llama.cpp returns it as `content` with
   `tool_calls: null`. opencode never executes the tool → the executor reports
   success it never performed. **Higher precision did not fix it** — it is a
   model/template behavior, not a quantization artifact. This eliminates the entire
   Qwen2.5-Coder family for the agentic loop, despite its excellent draft
   acceptance (84%).
2. **Qwen3 tool-calling is what llama.cpp parses** (`finish_reason: tool_calls`,
   native, JSON-escaped arguments). This is the family to use.
3. **But Qwen3-14B is too small to emit strict JSON reliably** — it produced
   trailing commas and unescaped inner quotes (`'edition = "2024"'`) that
   `serde_json` rejects. The plan/step/review envelopes need *valid* JSON.
4. **Qwen3-Coder-30B-A3B fixes both** — it is tuned for agentic tool use *and*
   structured output, so it emits native tool calls **and** clean JSON, and
   completes the loop. As an MoE with only ~3.3B active params it is also the
   **fastest** option measured (156 t/s decode), and it fits 24 GB whole.

## Why the winner runs *solo* (no speculative decoding)

The optimizations plan assumed a dense target + tiny draft with speculative
decoding. Spec-decode **does** work on this stack once correctly configured (see
the config note below) and is a real win on **dense** targets — measured here:

| Dense pair | Decode (no draft) | Decode (spec-decode) | Speedup | Draft accept |
|---|---|---|---|---|
| Qwen2.5-Coder-14B Q4 + 1.5B | ~76 t/s | **133 t/s** | ~1.75× | 84% |
| Qwen3-14B Q4 + 0.6B | ~76 t/s | **97 t/s** | ~1.29× | 55% |

But the model that actually **completes the loop** is an **MoE** (Qwen3-Coder-30B-
A3B). On an A3B MoE only ~3.3B params are active per token, so decode is already
fast (~156 t/s) and a draft cannot beat it — the public RTX 3090 benchmark of
speculative decoding on Qwen3 A3B finds **no net speedup**. So the winning config
runs the MoE **solo**: faster than either dense spec-decode pair *and* correct.

Net: speculative decoding is a useful lever for **dense** models that don't fit a
fast solo budget, but here a fast **MoE coder** beats it on both speed and loop
reliability.

## Configuration discoveries (the things that cost the most time)

These are stack-level gotchas, independent of the model choice:

1. **`--spec-type draft-simple` is REQUIRED to enable draft-model speculative
   decoding on llama.cpp b9728.** The new modular spec API defaults to
   `--spec-type none`; passing `--model-draft` alone loads the draft but engages
   nothing, logging `common_speculative_init: no implementations specified for
   speculative decoding` and silently running the target solo. With
   `--spec-type draft-simple` the log shows `adding speculative implementation
   'draft-simple'` and per-request `draft_n` / acceptance stats appear. (There are
   also `--fim-qwen-14b-spec` / `--spec-default` presets.)
2. **opencode must be current (1.17.x).** This box had **1.4.3**; the harness's
   file-based JSON hand-off and reliable tool execution were built for ~1.17.10.
   On 1.4.3 the agents intermittently failed to write/parse the envelope. Upgrade
   in place: `opencode upgrade` (1.4.3 → 1.17.11). Config (`opencode.json`) is
   untouched by the upgrade. **Do not pin an older opencode** to match old harness
   behavior — go forward.
3. **`--spec-type draft-simple` tolerates a 128-token vocab gap.** Qwen2.5-Coder's
   small drafts (0.5–3B) use vocab 151936 while 7–32B use 152064 — a 128-token
   difference. That gap is the documented tolerance limit, and `draft-simple`
   accepts it (it engaged fine for Coder-14B + 1.5B). The earlier "no
   implementations" error was the missing `--spec-type`, **not** the vocab gap.
4. **Qwen3 uses one 151936-token vocab across all sizes** (dense, MoE, and
   Qwen3-Coder), so any Qwen3 target pairs exactly with a Qwen3-0.6B draft — handy
   if you do want spec-decode on a dense Qwen3 target.
5. **opencode 1.x needs a zero `cost` block per model** or its cost-calc throws
   `DecimalError`. The `qwen-coder` entry carries `"cost": {"input":0,"output":0}`.

## Reproduce

```bash
# 1. Start the winning config (Qwen3-Coder-30B-A3B, solo, GPU-resident):
docs/config/nvidia-3090/start-qwen3-coder.sh          # -ngl 99, port 8080, alias qwen-coder

# 2. opencode.json: llamacpp/qwen-coder at http://127.0.0.1:8080/v1,
#    apiKey "local", zero "cost" block, limit.context 32768. Needs opencode >= 1.17.

# 3. Run the loop:
MODEL=llamacpp/qwen-coder ./scripts/demo-orchestrate.sh

# Spec-decode variants (dense targets) live in start-coder-specdecode.sh;
# they work but are slower than the MoE solo here.
```

## Highest-value next steps

- **Tighten the executor envelope.** Step 1 still returned `status: unknown`
  (the executor said "DONE" without the JSON envelope) but the reviewer continued;
  the same structured-output reliability item from the macOS PoC remains.
- **Liberal JSON parsing (done, partial).** The harness now retries a failed
  `serde_json` parse against a string-aware **trailing-comma stripper**
  (`llm::relax_json`, Postel's "be liberal in what you accept") — the harness owns
  this parse, so it is the right layer to relax. This fixes the **trailing-comma**
  failure. It deliberately does **not** repair **unescaped inner quotes** (e.g. a
  model writing `"4.0.0"` inside a string value) — that is ambiguous and any guess
  can silently corrupt the plan. Qwen3-14B emits *both* malformations, so it still
  isn't reliable; the clean-JSON 30B-A3B coder needs neither repair.
- **Constrain JSON at the source (the real fix for unescaped quotes).** llama.cpp
  `--grammar` / json-schema would make malformed JSON *impossible* ("be
  conservative in what you send"), but that constraint is injected at the request
  layer, which opencode owns — not reachable from the harness through this stack.
- **A `Hello, {name}!` nudge in the spec** if the literal greeting matters; the
  current spec's "a friendly greeting" is ambiguous enough that the model returned
  the bare name.
