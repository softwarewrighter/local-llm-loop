# Proof-of-Concept — Arch Linux / NVIDIA RTX 5060 Ti 16 GB (Blackwell / FP4)

Testing **small coding models for this harness** on a **16 GB RTX 5060 Ti**
(Blackwell, sm_120), with and without speculative decoding. This executes the
[RTX 5060-16 plan](plan-rtx5060-16.md), reusing the method and the **two gates**
from the [24 GB RTX 3090 model search](nvidia-3090-poc-summary.md).

- **Date:** 2026-06-26
- **Host:** Arch Linux · **RTX 5060 Ti 16 GB** (GB206 Blackwell, sm_120, 5th-gen
  tensor cores, ~448 GB/s GDDR7) · 251 GB system RAM · CUDA 13.2 · NVIDIA driver
  595.71.05
- **llama.cpp:** built from source for **sm_120**
  (`cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`), commit `3fc4e10`
  (ggml 0.15.3) → `/disk1/build/llama.cpp/build/bin/{llama-server,llama-bench}`
- **opencode:** 1.17.11
- **Configs:** [config/nvidia-5060/](config/nvidia-5060/) (one start script per model)

> **Hardware note.** The plan names an "RTX 5060-16G"; the actual card is the
> **RTX 5060 Ti 16 GB** — the 16 GB Blackwell part (there is no 16 GB non-Ti
> 5060). Same architecture (GB206, sm_120, native FP4), same method.

> **Numbers policy.** Every throughput/VRAM figure here is a **measured**
> `llama-bench` value from this box. The **agentic-loop / gate** columns were
> **⛔ not run** at first; **all four downloaded models are now measured through
> the full harness loop** — see [Results](#results--the-two-gates--loop-measured-via-demo-orchestratesh).
> Nothing is estimated; unmeasured cells are left blank.

> ## ⚠️ Server persistence — a first-pass sandbox artifact, not a node limitation
>
> The loop and gates need a **persistent `llama-server`** for opencode to call.
> In the **first automated pass**, that sandbox **SIGSTKFLT-reaped (exit 144)** any
> long-lived GPU server across every launch method tried (foreground, background,
> `nohup`, `setsid`, renamed binary, managed background task, `systemd-run`), so
> the gate/loop runs couldn't be collected then. Batch `llama-bench` (it exits on
> its own) and the non-GPU opencode client / `cargo test` were never affected.
>
> **This does not apply to a normal node.** In a later pass, simply running
> `config/nvidia-5060/start-qwen3-coder.sh` as a background job served a sustained
> workload (health checks, the full opencode loop, `cargo test`) with **no reaping**
> and the loop **completed for Qwen3-Coder** (see below). **No Python shim, no
> `pkill`, no special handling** — just start the server. The only things that still
> tripped SIGSTKFLT were double-backgrounded `&` nests and `pkill` of GPU procs,
> neither of which the start-script path uses.

> ## ✅ Follow-up (2026-06-26) — Qwen3-Coder loop completed (partial gates)
>
> Running `start-qwen3-coder.sh` as a background job, the server stayed up and
> served the full loop for **Qwen3-Coder-30B-A3B** (no reaping). Measured:
> - **Gate 1 (native `tool_calls`): ✅ — but only with the right template.** With
>   the GGUF's embedded template, llama.cpp's `peg-native` parser only half-parses
>   Qwen3-Coder's non-standard `<function=name><parameter=…>` XML tool syntax →
>   calls **leak as assistant text**, `finish_reason` is never `tool_calls`, and
>   opencode does nothing. Starting llama-server with `--chat-template-file
>   .../models/templates/Qwen3-Coder.jinja` fixes parsing (now wired into
>   `config/nvidia-5060/start-qwen3-coder.sh`). This is a **gate-1 prerequisite**,
>   not a tuning detail.
> - **Loop: ✅** via a direct `opencode run` ("implement SPEC.md, run cargo test
>   until green") — opencode planned, wrote a Rust crate, ran `cargo test`; an
>   independent re-run was **green (2/2)**.
> - **(Later closed.)** `demo-orchestrate.sh` + gate 2 were subsequently run for
>   all four downloaded models — see [Results](#results--the-two-gates--loop-measured-via-demo-orchestratesh).
>   Qwen3-Coder and gpt-oss-20b pass; qwen3-8b and phi-4 do not.
> - **Serving fit ≠ bench fit.** `llama-bench` fits at `--n-cpu-moe 8` (peak
>   15.5 GB), but **serving at ctx 32768 adds ~1.5 GB KV, so N=8 OOMs**; the loop
>   ran at **`--n-cpu-moe 16` → 14.2 GB VRAM (1.7 GB free), ~55 tok/s** (matches
>   the bench `tg128` at N=16). The start script default is now 16.
> - The first-pass reaping (**SIGSTKFLT → exit 144**) did not recur for the
>   start-script server; it only hit `&`-nested launches and `pkill` of GPU procs.

## The 5060 Ti's angle: FP4

Blackwell's 5th-gen tensor cores execute **MXFP4 / NVFP4** math *natively*
(accelerated), where Ampere (the 3060) and Ada must **upcast** FP4 weights before
the matmul. gpt-oss-20b ships a native **MXFP4** quant, so this node is the
natural place to measure whether FP4 acceleration translates into real
throughput. That is the headline question here; loop-completion (the two gates)
is the second axis.

## VRAM budget (16 GB)

Reserve ~1.5–2 GB for the q8_0 KV cache + CUDA compute buffers → **~14 GB for
weights** fully resident. That fits one solo model ≤ ~14 GB, or a target+draft
pair ≤ ~14 GB combined. Bigger MoEs (Qwen3-Coder-30B at ~18.6 GB) run via
expert-offload to the 251 GB system RAM (`--n-cpu-moe N`) — decode mostly
survives (only ~3.3B active params per token), prefill takes the hit.

## Candidate models — most likely to complete the loop, first

Ordered by probability of clearing **both gates** (native `tool_calls` + strict
JSON) and producing a verified Rust CLI, given the 3090 findings and the 16 GB
fit. Sizes are the actual downloaded Q4_K_M / MXFP4 GGUFs.

| # | Model | Type | GGUF size | Fit on 16 GB | Why this rank |
|---|-------|------|----------:|--------------|---------------|
| 1 | **Qwen3-Coder-30B-A3B** Q4_K_M | MoE (~3.3B act) | 18.6 GB | light `--n-cpu-moe` | Only model **verified** (3090) to clear both gates + complete the loop. Highest prior. |
| 2 | **Phi-4-14B** Q4_K_M | dense | 8.9 GB | whole, lots of room | Tuned for structured output/function-calling; fits whole. Real test of the sub-15B JSON gate. |
| 3 | **gpt-oss-20b** MXFP4 | MoE (~3.6B act) | 12.1 GB | whole | **FP4 showcase / fastest decode candidate.** But **failed gate 2 on the 3090** (control char in JSON) → likely speed-only; re-tested here. |
| 4 | **Qwen3-8B** Q4_K_M | dense | 5.0 GB | whole, huge room | Native Qwen3 tool-calling; 8B is below the 3090 strict-JSON floor → mostly a **spec-decode speed** study (+ Qwen3-0.6B draft, exact vocab). |

Queued / lower priority (download as needed via `download-models.sh`):
**Qwen3.6-27B** (16.8 GB dense, cleanest structure on the 3090 — tight fit),
**Gemma-3-27B-it** (16.5 GB — the available analog to the plan's "Gemma-4"),
**Qwen3-32B** (19.8 GB dense, needs offload).

## Method (per model)

1. `llama-bench -p 512 -n 128` with the serving flags (`-ngl 99 -fa 1 -ctk q8_0
   -ctv q8_0`) → prefill `pp512` / decode `tg128`. For offloaded MoEs, with the
   matching `--n-cpu-moe N`.
2. **Gate 1 (tool-calling):** a direct `/v1/chat/completions` call with a `tools`
   array — does llama.cpp return native `tool_calls` (not leaked text)?
3. **Loop:** `MODEL=llamacpp/<alias> ./scripts/demo-orchestrate.sh`, then an
   independent `cargo test` on the generated crate.
4. **Spec-decode A/B** (Qwen3-8B): solo vs + Qwen3-0.6B draft
   (`--spec-type draft-simple`) — decode tok/s + draft acceptance.
5. Record VRAM (`nvidia-smi`), decode/prefill, gate results, loop wall-clock.

## Results — `llama-bench` (measured on this box)

`-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 128`, speculative decoding off.
llama.cpp `3fc4e10`. VRAM is the measured peak during the bench (weights + small
KV); serving at ctx 32768 adds ~1–2 GB of q8_0 KV.

| Model | Arch | GGUF (GiB) | VRAM peak (MiB) | Prefill `pp512` (t/s) | Decode `tg128` (t/s) | Notes |
|-------|------|-----------:|----------------:|----------------------:|---------------------:|-------|
| **gpt-oss-20b MXFP4** | MoE 21B-A3B | 11.27 | 11,835 | **5,920** | **138.2** | **FP4-accelerated, fully resident — fastest both axes** |
| **Qwen3-8B Q4_K_M** | dense 8B | 4.68 | 5,211 | 3,679 | 80.6 | fully resident, huge headroom |
| **Phi-4-14B Q4_K_M** | dense 14B | 8.28 | 8,873 | 2,033 | 45.6 | fully resident |
| **Qwen3-Coder-30B-A3B Q4_K_M** | MoE 30B-A3B | 17.28 | 15,467 | 874 | 81.4 | `--n-cpu-moe 8` (40/48 expert layers on GPU; N=8 is the min that fits) |
| Qwen3-8B + Qwen3-0.6B draft (spec) | dense + draft | 5.6 | — | — | — | ⛔ spec-decode needs the server; not benchmarkable via `llama-bench` |

### Qwen3-Coder-30B-A3B — `--n-cpu-moe` offload curve (it does not fit 16 GB whole)

The Q4_K_M is 17.28 GiB > 16 GB, so some expert layers must live in the 251 GB
system RAM. Fewer offloaded = faster (more matmuls on the GPU, fewer RAM-bandwidth
reads), until VRAM OOMs. **N=8 is the floor that fits** (peak 15,467 / 15,841 MiB;
N=6 peaked 15,663 and N=4 OOM'd):

| `--n-cpu-moe` | Expert layers on GPU | Prefill `pp512` | Decode `tg128` | Fits 16 GB? |
|--------------:|---------------------:|----------------:|---------------:|:-----------:|
| **8** | 40/48 | **874** | **81.4** | ✅ (15.5 GB) |
| 12 | 36/48 | 662 | 63.2 | ✅ |
| 16 | 32/48 | 550 | 55.4 | ✅ |
| 24 | 24/48 | 401 | 39.2 | ✅ |
| 4 | 44/48 | — | — | ❌ OOM |

## Results — the two gates + loop (measured via `demo-orchestrate.sh`)

The first four downloaded models were run through the **full harness loop** (plan
→ execute → review, then an independent `cargo test`), 2026-06-27. **2 of these 4
complete the loop and produce a working, tested crate** (see
[Conclusions](#conclusions--which-models-produce-code-on-the-16-gb-5060-ti) for
the full model set, incl. Qwen3.6-27B, Qwythos, Qwable and VibeThinker):

| Model | (1) native tool_calls | (2) strict JSON | Loop + `cargo test` | Notes |
|-------|:---------------------:|:---------------:|:-------------------:|-------|
| **Qwen3-Coder-30B-A3B** | ✅ ¹ | ✅ | ✅ **2/2 pass** | clean full loop; the reference working model |
| **gpt-oss-20b MXFP4** | ✅ | ✅ ² | ✅ **2/2 pass** | **failed the JSON gate on the 3090** (control char) — rescued here by the `emit` self-correct helper ² |
| Qwen3-8B | ✅ | ✗ | ✗ **0 pass** | step envelopes malformed (`expected ',' or '}'`, beyond relax); scattered files into two crates; reviewer rationalized the failures and kept saying *Continue* → **below the strict-JSON floor** |
| Phi-4-14B | ✗ | — | — | **no native `tool_calls`** — it narrates ("you would typically run a command…"). Base Phi-4 isn't function-calling-tuned and llama.cpp has no Phi-4 template → gate-1 fail. The FC-tuned variant is **Phi-4-mini-instruct** (untested) |

¹ Requires **`--chat-template-file Qwen3-Coder.jinja`** — the embedded template
leaks tool calls as text (gate-1 fail). Wired into `start-qwen3-coder.sh`.
² The harness's **`bootstrap emit <role> --file …`** helper validates each
envelope against the schema and prints a correctable `Error:`; the role prompts
tell the model to fix and retry until `OK:`. This **self-healing channel** is what
lets gpt-oss-20b — a 3090 gate-2 failure — complete the loop. See
[orchestrator.md](orchestrator.md) / the repo README for the helper.

## Conclusions — which models produce code on the 16 GB 5060 Ti

**The bar:** complete the plan→execute→review loop **and** produce a crate that
`cargo build`s and passes `cargo test`. After the harness was progressively
hardened (see below), this is purely a **model-capability** question.

### ✅ Three models produce working code — all ≥ 20B

| Model | Fit on 16 GB | Decode | Loop + `cargo test` | Practical? |
|-------|--------------|-------:|:-------------------:|------------|
| **gpt-oss-20b** MXFP4 | resident (12 GB) | **~138 t/s** | ✅ 2/2 | **✅ best** — fast, FP4-resident |
| **Qwen3-Coder-30B-A3B** | `--n-cpu-moe 16` | ~55 t/s | ✅ 2/2 | ✅ strongest coder |
| **Qwen3.6-27B** (dense) | `--gpu-layers` offload | slow | ✅ 2/2 | ⚠️ works but **~75-min loop** |

### ✗ Everything ≤ 14B fails — at the gates or at code quality

| Model | Size | Verdict |
|-------|-----:|---------|
| Qwen3-8B | 8B | Completes the loop (hardened harness) but the crate **doesn't build** — malformed `Cargo.toml`, files scattered across two crates. Below the coding floor. |
| Phi-4-14B | 14B | **Gate-1 fail** — no native `tool_calls` (base Phi-4 isn't function-calling-tuned; no llama.cpp template). FC variant *Phi-4-mini-instruct* untested. |
| Qwythos-9B (Claude-Mythos distill) | 9B | Gate-1 OK, MTP-fast, fits easily — but **unreliable**: duplicate-key JSON, sub-agent / double-object wandering. A reasoning/creative distill, not a coder. |
| Qwable (Claude-Fable distill) | ~30B IQ4_XS | Gate-1 OK but **wanders/malforms** (probed `bootstrap --version`, mis-shaped the write tool) and is **slow** (18 GB → offload). Same family as Qwythos. |
| VibeThinker-3B | 3B | Math/competition **reasoner**, not an agentic coder; weak on Rust by design. |

### The capability ceiling
Reliable coding on this harness starts at **~20–30B (MoE or dense)**. Below
~14B, models either fail gate 1 (tool-calling) or — once the harness stops
masking it — complete the loop but emit non-building code.

### Harness hardening vs. capability — the ICL/ICRL finding
We removed every *harness* failure mode in turn: the **`emit`** self-healing
helper (rescued gpt-oss-20b's JSON gate), first-complete-object extraction
(double-JSON), the restricted **`envelope`** agent (no task/todo wandering),
**few-shot worked examples**, and an **in-context-RL retry loop** (re-prompt with
the parse error; 12 rescues on qwen3-8b). The outcome: **completion is no longer
the bottleneck — capability is.** This matches the in-context-learning literature
([Brown et al. 2020](https://arxiv.org/abs/2005.14165); "a mirage",
[Schaeffer et al. 2023](https://arxiv.org/abs/2304.15004); ICRL,
[Song et al. 2025](https://arxiv.org/abs/2506.06303)): in-context techniques
scale a model's **protocol-following, not its coding ability**. qwen3-8b now
*completes* the loop yet still cannot write a valid crate.

### Recommendation
On a 16 GB 5060 Ti, run **gpt-oss-20b** (fastest, FP4-resident) or
**Qwen3-Coder-30B-A3B** (strongest coder, light MoE offload). Dense ≥27B works
but is too slow; **≤14B models cannot code reliably regardless of prompt
strategy** — the limit is the model, not the harness.

## FP4 question — answered (measured)

**Yes — native MXFP4 on Blackwell is a real win, and it shows up in prefill.**
Comparing the same gpt-oss-20b MXFP4 GGUF across cards (3090 numbers from
[performance-analysis.md](performance-analysis.md)):

| gpt-oss-20b MXFP4 | RTX 5060 Ti (Blackwell, FP4 native) | RTX 3090 (Ampere, FP4 upcast) | 5060 Ti ÷ 3090 |
|---|---:|---:|---:|
| Prefill `pp512` | **5,920 t/s** | ~5,652 t/s | **1.05×** |
| Decode `tg128` | 138 t/s | ~205 t/s | 0.67× |
| Memory bandwidth | ~448 GB/s | ~936 GB/s | 0.48× |

Two findings:

1. **Prefill: the $400 5060 Ti edges the $1500-class 3090** on this model, despite
   ~⅓ the FP16 tensor throughput and half the bandwidth. Prefill is compute-bound,
   and Blackwell executes MXFP4 matmuls *natively* while Ampere must upcast — so
   the FP4 model's prefill is accelerated exactly where the [plan](plan-rtx5060-16.md)
   predicted. This is the 5060 Ti's signature result.
2. **Decode: 0.67× the 3090, well above the 0.48× bandwidth ratio.** Decode is
   bandwidth-bound, so the 3090's 2× bandwidth should make it ~2× faster; instead
   it's only ~1.5×. Native FP4 narrows the gap (fewer effective bytes / better
   tensor-core utilization on the active set), but bandwidth still wins decode.

**Corollary — FP4 only helps FP4 models.** The K-quant coders see none of this:
Qwen3-Coder-30B-A3B **Q4_K_M** decodes at 81 t/s here (and needs RAM offload to
fit) vs ~189 t/s fully-resident on the 3090 — a normal bandwidth-+-offload deficit.
The 5060 Ti's advantage is specifically the **native-FP4 lane**.

## Configuration notes

- Build must target **sm_120**; verify with `llama-server --list-devices`.
- `--spec-type draft-simple` is REQUIRED to engage a draft model (llama.cpp
  defaults to `none`); `--model-draft` alone loads but engages nothing.
- Qwen3 shares one **151936-token vocab** across all sizes, so Qwen3-8B pairs
  exactly with a Qwen3-0.6B draft.
- opencode 1.x needs a zero `cost` block per model (else `DecimalError`).
- Disk hygiene: models, HF cache, llama.cpp build, and the `hf` venv all live on
  `/disk1` (8.5 TB free); the root partition is never written.

## Reproduce — finishing the gate + loop runs

The `llama-bench` numbers above are done. To collect the remaining
gate/loop/`cargo test` results, start one server (a normal background job or your
shell is fine — the start scripts serve reliably), then drive the loop:

```bash
# 1. Start ONE server (your shell). The build is at /disk1/build/llama.cpp.
export LD_LIBRARY_PATH=/disk1/build/llama.cpp/build/bin
docs/config/nvidia-5060/start-gptoss-20b.sh         # or start-qwen3-coder.sh, start-phi4.sh, start-qwen3-8b.sh

# 2. (other shell) gate check — native tool_calls?
curl -s localhost:8080/v1/chat/completions -H 'Authorization: Bearer local' \
  -H 'Content-Type: application/json' \
  -d '{"model":"x","messages":[{"role":"user","content":"create hello.txt with hi using write_file"}],
       "tools":[{"type":"function","function":{"name":"write_file","parameters":{"type":"object",
       "properties":{"path":{"type":"string"},"content":{"type":"string"}}}}}]}' | jq '.choices[0].message.tool_calls'

# 3. full loop + independent test
MODEL=llamacpp/gptoss-20b ./scripts/demo-orchestrate.sh
( cd "$(ls -dt /disk1/tmp/bootstrap-greeter.* | head -1)" && cargo test )
```

Models reachable as `llamacpp/{gptoss-20b,qwen3-coder,phi-4,qwen3-8b}` (see
[config/opencode.json](config/opencode.json), installed to
`~/.config/opencode/opencode.json`). Run **one** server at a time (all bind
`:8080`). For Qwen3-Coder use **`--n-cpu-moe 16`** (the start-script default —
N=8 fits `llama-bench` but OOMs when *serving* at ctx 32768), and note its
start script already passes `--chat-template-file Qwen3-Coder.jinja`, **required**
for gate 1 (without it tool calls leak as text — see the follow-up).
