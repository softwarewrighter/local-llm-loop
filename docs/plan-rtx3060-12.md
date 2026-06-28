# Plan — small-coder testing on an RTX 3060-12G node

How to test **coding models for this harness** on an **RTX 3060 12 GB** node,
**preferring MTP** (self-speculative decoding). This is a *plan*; it builds on the
[12 GB RTX 3060 CPU-offload POC](arch-nvidia-3060-poc-summary.md) (Qwable, MoE
experts in RAM) and folds in the later model-capability findings from the
[16 GB RTX 5060 Ti POC](nvidia-5060-poc-summary.md) and the
[Ornith family research](plan-ornith-models.md). Start-script templates:
[config/arch-nvidia-3060/](config/arch-nvidia-3060/).

> **Goal.** Identify **3 models** that (1) **fit** in 12 GB, (2) **produce
> correct code** (clear both gates + `cargo test` green), and (3) run
> **relatively fast** — **preferably with MTP**. Once 3 are confirmed, a separate
> effort compares *code quality* across tasks (TBD).

## What the prior boxes already settled (constrains the candidate list)

Three results from the 5060/3090/M1 work pin down this plan — the original
candidate list (Phi-4-14B, Qwen3-8B as loop-completers) is **superseded**:

1. **Capability floor is ~20–30B.** Every *general* model ≤14B failed — Qwen3-8B
   (emits non-building code), Phi-4-14B (gate-1: no native `tool_calls`),
   Qwythos-9B (malformed JSON). Reliable coding starts at ~20–30B (MoE or dense).
2. **Ornith-1.0-9B is the one sub-14B exception** — RL-trained for agentic coding
   (SWE-bench Verified 69.4). On the 5060 it ran the full loop in **4m42s, 0
   retries**, producing a clean crate. It fits 12 GB *whole*.
3. **Ampere has no native FP4** — it upcasts MXFP4. So gpt-oss-20b and
   Gemma-4-26B (the Blackwell FP4 winners) lose their edge here; **K-quants are
   preferred** on this box.

## Hardware profile

| | RTX 3060-12G (the dual-Xeon box) |
|---|---|
| Arch | **Ampere** (GA106), sm_86, 3rd-gen tensor cores |
| VRAM | **12 GB GDDR6**, ~360 GB/s |
| Host RAM | **503 GB** DDR4 (dual Xeon E5-2697 v4) — enables MoE expert-offload |
| FP4 | **not** hardware-accelerated (Ampere upcasts FP4/MXFP4) → prefer K-quants |

**Build:** reuse the working CUDA build (sm_86, b9784+) from the 3060 POC. Confirm
`--spec-type draft-mtp` exists on the build (needed for the MTP candidate).

**The 3060's two regimes:** small VRAM but **huge system RAM**.
- **Dense model that fits 12 GB** → fully GPU-resident, fast (Ornith-9B).
- **MoE too big for 12 GB** → experts in the 503 GB RAM (`--n-cpu-moe N`),
  dense/attention + KV on the GPU — the proven Qwable-on-3060 trick. Decode
  survives (only ~active params read); **prefill takes the hit**.

## VRAM budget (12 GB)

Reserve ~1.5 GB for the q8_0 KV + compute → **~10–10.5 GB for resident weights**,
or tune an offloaded MoE to ~11 GB VRAM via `--n-cpu-moe`.

## The MTP angle (why only one of the three uses it)

MTP (self-speculation via an embedded next-token head, `--spec-type draft-mtp`)
gives a free decode uplift — **but only where it fits and helps**:
- **Ornith-9B (dense):** no MTP head, and no faster same-vocab draft (custom
  248320 vocab) → **runs solo**, no MTP gain on 12 GB.
- **Qwen3-Coder-30B-A3B:** A3B MoE, no published MTP head → **solo**.
- **Qwen3.6-35B-A3B-MTP:** A3B MoE *with* an MTP head → **the MTP pick.**
  Offloaded to RAM, decode rides only ~3B active params and MTP recovers what the
  spill costs (~85% accept, ~1.24× on the M1 Max). Same offloaded-MoE regime
  Qwable already completed the loop in here, plus a real coder + the head.

## The 3 models to test (ranked by confidence: fit + codes + fast)

| # | Model | Quant / size | Placement on 12 GB | MTP | Start script |
|---|-------|--------------|--------------------|-----|--------------|
| **1** | **Ornith-1.0-9B** | Q6_K (7.36 GB) | **resident** (Q8_0 9.53 GB = stretch) | — (dense) | [`start-ornith-9b.sh`](config/arch-nvidia-3060/start-ornith-9b.sh) |
| **2** | **Qwen3.6-35B-A3B-MTP** | Q4_K_M (~21 GB) | `--n-cpu-moe` offload | **✅ `draft-mtp`** | [`start-qwen36-mtp.sh`](config/arch-nvidia-3060/start-qwen36-mtp.sh) |
| **3** | **Qwen3-Coder-30B-A3B** | Q4_K_M (~17.3 GB) | `--n-cpu-moe` offload | — | [`start-qwen3-coder.sh`](config/arch-nvidia-3060/start-qwen3-coder.sh) |

- **#1 Ornith-9B** — the anchor: highest-confidence "fits + codes", resident, fast
  (terse, 0 retries on the 5060). The only sub-14B that codes.
- **#2 Qwen3.6-MTP** — satisfies the MTP preference; proven offloaded-MoE regime
  on this exact box; A/B the MTP uplift with `SPEC_OFF=1`.
- **#3 Qwen3-Coder** — the strongest, most-verified coder (both gates + loop on
  3090 *and* 5060). No MTP, but the reliable workhorse. **Needs
  `--chat-template-file Qwen3-Coder.jinja`** or gate 1 fails (tool calls leak as
  text — already wired into the start script).

**Stretch / 4th if time:** **gpt-oss-20b** MXFP4 (12.1 GB, fits whole but **slow
on Ampere** — FP4 upcast; a speed-ceiling data point only) and **Ornith-1.0-35B
APEX-MTP** (26.2 GB GGUF, offloaded — a *coder* with MTP, but the head isn't in
official weights, so verify it loads first).

## Fallback tier — if the offloaded MoEs run too slow (> 10 min loop)

If #2/#3 blow past ~10 min (offload prefill is RAM-bound on this box), **don't
reach for a draft pair on another MoE** — reach for a **resident dense coder with
MTP self-spec**. Two facts decide this:

- **Spec-decode rewards *dense* models, not A3B MoEs.** A draft/MTP amortizes the
  weights read per token; an A3B MoE already reads only ~3B active params, so on
  **Ampere it's break-even-to-negative** — measured **−40% to −52%** on a 3090
  ([thc1006 Qwen3.6-A3B benchmark](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090)).
  So MTP will *not* rescue the offloaded MoEs here.
- **The only sub-14B that actually codes is Ornith-9B** (dense), and it now ships
  with a baked-in MTP head — so the fast fallback is *resident*, not offloaded.

| Model | Target+draft size | Spec route | Codes? (gate status) | Verdict |
|-------|-------------------|------------|----------------------|---------|
| **Ornith-1.0-9B + MTP head** ([`start-ornith-9b-mtp.sh`](config/arch-nvidia-3060/start-ornith-9b-mtp.sh)) | 5.6–7.4 GB, self-spec | **MTP (`draft-mtp`)** | ✅ proven coder (SWE 69.4, 0 retries) | **Best fallback** — resident, **~1.4–1.7×** decode (A6000, Ampere), no offload |
| Qwen2.5-Coder-7B + 0.5B/1.5B draft | ~5–7 GB | exact vocab (151936), **2.5× measured** | ❌ **gate-1 fail** — tool-calls broken in llama.cpp | speed study only |
| Qwen3-14B + Qwen3-0.6B draft | ~9.5 GB | exact vocab (151936) | ⚠️ general ≤14B floor (Qwen3-8B failed) | risky; largest general dense that fits w/ draft |
| Qwen3-8B + Qwen3-0.6B draft | ~6 GB | exact vocab (151936) | ❌ emits non-building code | speed study only |
| Ornith-9B + external draft | — | ✗ custom **248320** vocab → no compatible draft | — | **no draft exists** → MTP is the only route |

**The MTP GGUF:** [`protoLabsAI/Ornith-1.0-9B-MTP-GGUF`](https://huggingface.co/protoLabsAI/Ornith-1.0-9B-MTP-GGUF)
bakes a KL-distilled next-token head onto the 9B trunk for `--spec-type draft-mtp`
— measured **~1.4–1.7× decode** on an A6000 (Ampere), acceptance **~0.766**,
distribution-lossless. Needs llama.cpp ≥ b9616 (the box's b9784+ build qualifies).
A/B the uplift with `SPEC_OFF=1`. Net: the fallback path that is **both fast
(spec-decode) and actually codes** is **Ornith-9B-MTP** — a separate draft model
is neither available (custom vocab) nor useful here.

## Test procedure (per model)

Reuse the 3060 CUDA build; all three scripts default to **PORT 8081** (8080 is the
opencode UI Docker stack here) and the opencode aliases already exist in
[config/opencode.json](config/opencode.json) (`ornith-9b-gguf`, `qwen36-mtp`,
`qwen3-coder`) — point that file's `baseURL` at `http://localhost:8081/v1`.

1. **Throughput** — `llama-bench -p 512 -n 128` with the serving flags
   (`-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0`, plus the matching `--n-cpu-moe N` for the
   offloaded MoEs) → prefill `pp512` / decode `tg128`. For #2, A/B with
   `SPEC_OFF=1` to measure the MTP uplift + draft acceptance.
2. **Gate 1 (tool-calling)** — a direct `/v1/chat/completions` call with a `tools`
   array; confirm native `tool_calls` (not leaked text). For #3 this requires the
   `Qwen3-Coder.jinja` template.
3. **Loop** — `MODEL=llamacpp/<alias> ./scripts/demo-orchestrate.sh`, then an
   **independent `cargo test`** on the generated crate (don't trust the model's
   claim).
4. **Record** VRAM (`nvidia-smi`), decode/prefill, retries, **loop wall-clock**,
   and pass/fail. Loop time ≈ *decode × tokens × (1 + retries) × steps*.

**Success bar:** clears both gates **and** `cargo test` green. Three green = the
candidate set for the (later) cross-task code-quality comparison.

## `--n-cpu-moe` tuning (the offloaded MoEs)

Qwable's IQ4_XS (17.6 GB) sat at ~11.2 GB VRAM with **N=18** (~413 t/s prefill /
~49 t/s decode). The two coders here are bigger Q4_K_M files, so **start higher
(N=24)** and lower N until 12 GB is near-full (more experts on GPU = faster). The
MTP head (#2) adds a little VRAM — leave margin. On this dual-socket box, if
decode is RAM-bandwidth-starved, try `numactl --interleave=all`.

## Open questions to answer with the runs

- Smallest/ fastest model that **completes the loop** with passing tests on 12 GB
  (expectation: Ornith-9B is both smallest and likely fastest end-to-end).
- **Does the MTP head load + help** under `--n-cpu-moe` offload on Ampere, and by
  how much (the `SPEC_OFF` A/B)? Decode here is more bandwidth-bound than the
  3090, so drafts may pay off more.
- The `--n-cpu-moe` curve (prefill/decode vs VRAM) for each offloaded coder —
  extend the Qwable N=18 datapoint.
