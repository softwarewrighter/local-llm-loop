# Plan — small-coder testing on an RTX 5060-16G node

How to test **small coding models for this harness** on an **RTX 5060 16 GB**
node, **with and without speculative decoding**. This plan is now **partially
executed** — measured results are in
[nvidia-5060-poc-summary.md](nvidia-5060-poc-summary.md); see *Status & next
steps* below. It reuses the method and hard-won rules from the 24 GB RTX 3090
work — see [nvidia-3090-poc-summary.md](nvidia-3090-poc-summary.md) (the
model-search matrix and the two gates),
[performance-analysis.md](performance-analysis.md), and
[config/nvidia-3090/](config/nvidia-3090/) for the start-script templates to copy.

## Status & next steps (2026-06-26)

**Done** (see [nvidia-5060-poc-summary.md](nvidia-5060-poc-summary.md)):
- sm_120 llama.cpp build; `llama-bench` matrix across the four resident candidates.
- **FP4 question answered:** native MXFP4 prefill on gpt-oss-20b **beats the 3090**
  (5,920 vs ~5,652 t/s `pp512`); decode 0.67× (bandwidth-bound). FP4 only helps
  FP4 models — the K-quant coders see none of it.
- **Qwen3-Coder-30B-A3B:** gate 1 (native `tool_calls`) ✅ and a full `opencode run`
  loop ✅ (`cargo test` green), serving at `--n-cpu-moe 16` (14.2 GB, ~55 tok/s).

**Key learning — tool-call templates are per-model and gate-1-critical.**
Qwen3-Coder needed `--chat-template-file Qwen3-Coder.jinja`; without it llama.cpp's
generic `peg-native` parser only half-parses its `<function=…>` XML tool syntax and
the calls **leak as assistant text** (gate-1 fail). **Check every new model** for
the same — treat "tools leak as text" as a template problem, not a model failure.
gpt-oss has `openai-gpt-oss-120b.jinja`; Phi-4 may need its own.

**Next steps, ordered:**
1. **Close qwen3-coder gate 2** — run the harness's own
   `scripts/demo-orchestrate.sh` (strict-JSON plan/step/review envelopes) from an
   **interactive shell** (dodges the SIGSTKFLT → exit-144 server reaping seen in
   the sandbox), then independently `cargo test`.
2. **Gate-test the other three resident models** — phi-4, gpt-oss-20b, qwen3-8b:
   native `tool_calls` (with the correct template) + strict JSON. gpt-oss failed
   gate 2 on the 3090 (control char) — see if a template/quant change fixes it.
3. **Spec-decode A/B (qwen3-8b)** — solo vs + Qwen3-0.6B draft
   (`--spec-type draft-simple`, exact 151936 vocab): decode tok/s + draft acceptance.
4. **Push borderline models over gate 2** with higher-precision quants
   (Q5_K_M / Q8 / UD-Q4_K_XL) — the cheapest lever; relevant for phi-4 / gpt-oss.
5. **Queued larger models** (via `config/nvidia-5060/download-models.sh`):
   Qwen3.6-27B, Gemma-3-27B-it, DeepSeek-Coder-V2-Lite-16B, Qwen3-32B (needs offload).
6. **Server persistence** — always launch from an interactive shell, or investigate
   the sandbox SIGSTKFLT reaping, so the loop can be automated end-to-end.
7. **Energy/task at ~145 W** — measure perf/W vs the 3090/3060.

## Hardware profile

| | RTX 5060-16G |
|---|---|
| Arch | **Blackwell** (GB206), sm_120, 5th-gen tensor cores |
| VRAM | **16 GB GDDR7**, 128-bit, **~448 GB/s** |
| Special | **native FP4/FP8** tensor math (MXFP4/NVFP4 run *accelerated*, not upcast) |
| Power | ~145 W (very power-efficient per token) |

**Build note:** the system `llama-server` must be a CUDA build for **sm_120**
(`cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`). An Ampere/Ada build will
not run on Blackwell. Verify with `llama-server --list-devices`.

**The 5060's angle:** it's the FP4 box. gpt-oss's native **MXFP4** and emerging
**NVFP4** coder quants decode with hardware acceleration here, where Ampere (the
3060) has to upcast them. So this node is the natural place to test FP4 quants.

## VRAM budget (16 GB)

Reserve ~1.5–2 GB for the KV cache (q8_0) + CUDA compute buffers. That leaves
**~14 GB for weights** when running fully GPU-resident — either one solo model, or
a **target + draft combined ≤ ~14 GB** for speculative decoding. Bigger MoEs can
still run via expert-offload to system RAM (`--n-cpu-moe`), at a prefill cost.

## Candidate models

### A. Solo (fully GPU-resident) — fast path

| Model | Type | ~Q4 size | Notes |
|---|---|---:|---|
| **gpt-oss-20b** MXFP4 | MoE | ~11.3 GB | **FP4-accelerated on Blackwell** — fastest decode candidate here. (Failed the JSON gate on the 3090; re-test, the 5060 doesn't change that but a higher-precision quant might.) |
| **Phi-4-14B** | dense | ~9 GB | Strong structured-output/function-calling; fits with room. Universal small pick. |
| **DeepSeek-Coder-V2-Lite-16B** | MoE (~2.4B act) | ~10 GB | "Feels light" — MoE, fits; verify both gates. |
| **Qwen3-8B** | dense | ~5 GB | Tool-calling OK; small → JSON-gate risk. |
| **Gemma-4-E4B** | dense (small) | ~4 GB | Gemma 4 tool-calling + structured JSON; lots of headroom. |
| Qwen2.5-Coder-14B | dense | ~9 GB | Best small *coder*, **but tool-calling is broken in llama.cpp** (see 3090 POC) → use only for raw-speed/quality benchmarking, not the agentic loop. |
| CodeGemma-7B / OpenCoder-8B | dense | ~5 GB | Coder alternatives; verify gates. |

### B. Speculative-decoding pairs (target + draft both in 16 GB)

All need `--spec-type draft-simple` to engage (llama.cpp defaults to none) and an
**exact/compatible vocab** between target and draft.

| Target + draft | Combined | Vocab | Notes |
|---|---:|---|---|
| **Qwen3-8B + Qwen3-0.6B** | ~6 GB | exact (151936) | clean pair, tons of headroom |
| **Qwen3-14B + Qwen3-0.6B** | ~10 GB | exact | dense 14B target, draft helps |
| **Gemma-4-E4B + Gemma-4-E2B** | ~6 GB | exact (262144) | all-Gemma small pair |
| Qwen2.5-Coder-14B + 1.5B/0.5B | ~10 GB | 128-gap (tolerated) | best draft acceptance (~84%) but **tool-calling broken** → speed test only |

## The two gates (carry over from the 3090 eval)

A model only completes the loop if it (1) emits **native `tool_calls`** llama.cpp
parses, and (2) emits **strict JSON** for the envelopes. **Small models flunk gate
2 more often** (we saw Qwen3-14B and Granite-8B fail on trailing commas / unescaped
quotes / control chars). So expect: small models are fine for **raw spec-decode
speed benchmarking**, but fewer will **complete the agentic loop**. The harness's
`relax_json` tolerates trailing commas only; unescaped quotes/control chars still
fail. Higher-precision quants (Q5/Q8, or UD-Q4_K_XL) are the cheapest lever to push
a borderline model over gate 2.

## Test procedure

1. Build llama.cpp CUDA **sm_120**; confirm the 5060 is seen and `--spec-type` /
   `--model-draft` flags exist.
2. Per model: copy a `config/nvidia-3090/start-*.sh` script, set its own `--alias`
   + GGUF path + `PORT 8080`, `--cache-type-k/v q8_0 --flash-attn on --jinja`.
3. Add a distinct `opencode.json` entry per alias (zero `cost` block; opencode ≥ 1.17).
4. **Gate test** (direct API): does `/v1/chat/completions` with a `tools` array
   return native `tool_calls` (not leaked text)?
5. **Loop test:** `MODEL=llamacpp/<alias> ./scripts/demo-orchestrate.sh`, then
   independently `cargo test` the generated crate.
6. **Spec-decode A/B:** run the same target solo vs target+draft; record decode
   tok/s, draft acceptance, and loop wall-clock.

## What to measure / open questions

- Decode + prefill (`llama-bench -p 512 -n 128`) solo vs spec-decode; draft
  acceptance per pair.
- Does **FP4 acceleration** make gpt-oss-20b / NVFP4 coders meaningfully faster
  here than on Ampere? (Compare to the 3090/3060 numbers.)
- Which small models clear **both gates** (the real working set for 16 GB)?
- Energy/task at 145 W — the 5060 may be the most power-efficient *capable* node.
