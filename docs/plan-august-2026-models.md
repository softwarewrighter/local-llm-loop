# August 2026 local-model evaluation plan

Three newly available models are next in line for this repository's standard
agentic-coding evaluation. The primary target is the **M1 Max with 64 GB unified
memory**. At least one candidate will later be repeated on the **RTX 3090 with
24 GB VRAM**; no new result is claimed until a complete harness run finishes.

- **Status date:** 2026-08-15
- **Workload:** `examples/spec-greeter.txt`, using the same plan -> execute ->
  review loop as the existing result tables
- **Success:** native tool calls, valid JSON envelopes (including recorded
  `emit` recovery), completed loop, and independently green `cargo test`
- **Record:** quant and runtime revisions, launch flags, resident memory,
  prefill/decode, wall-clock, steps, retries, and final test count

## Candidates and current state

| Candidate | Architecture | Local state | M1 Max 64 GB | RTX 3090 24 GB |
|-----------|--------------|-------------|----------------|----------------|
| **Laguna S 2.1** | 118B total / 8B active MoE | **Measured failure:** tool gate and plan pass; full loop stopped after 13m03s in step 2 with no progress and non-compiling code | Fits at 16k; ~28 t/s decode, but this quant/run is not agent-efficient | Current 41 GB quant cannot fit wholly in VRAM; offload-only, low priority |
| **NVIDIA Nemotron 3.5 Lightning 30B-A3B** | 30B total / 3B active hybrid MoE (23 Mamba, 6 attention, 23 MoE blocks) | **Measured gate failure:** official ggml-org Q4_0 loads and passes native tool use, but fails 3/3 plan attempts; an Unsloth GGUF was also rejected as malformed | Fits whole at 32k; ~56-62 t/s decode, but fails this harness's plan protocol | 18.90 GB target should fit with conservative context; loop quality is already disqualifying unless prompting changes |
| **Qwen3.8-27B** | 27B dense | Released 2026-08-14; not downloaded or tested here | Expected to fit as a 4-6 bit quant; dense decode may be slow on Metal | Quant + KV must stay under 24 GB; target roughly 18-20 GB weights for headroom |

The local cache says **Laguna S 2.1**, not “Laguna S 5.1.” Both the Hugging Face
repository and downloaded filename identify version 2.1. Use that exact name in
results so the run is reproducible.

Nemotron setup and format research already lives in `~/tools/nemotron-flash`.
That work establishes why the native NVIDIA NVFP4/FP8 checkpoint is unsuitable
for this M1 Max and supplies recommended llama.cpp launchers for both machines.
The repository evaluation should consume those launchers rather than duplicate
or silently diverge from them.

## Test order

1. **Qwen3.8-27B / M1 Max** -- choose one reproducible GGUF, record its source
   revision and size, and compare this dense model with Qwopus3.6-27B. Start with
   GGUF for cross-hardware parity; add MLX only as a separately labeled run.
2. **Nemotron follow-up (optional)** -- retry only after changing the agent
   prompt/tool policy. The official GGUF and runtime work; the model repeatedly
   chooses unrelated filesystem inspection instead of the required plan emit.
3. **Laguna follow-up (optional)** -- retry only with a tested upstream/poolside
   runtime change or a tighter reasoning/tool-turn policy. The first 16k run is
   already sufficient to reject this setup as a practical default.
4. **RTX 3090 follow-up** -- start with whichever of Qwen3.8 or Nemotron has the
   best M1 Max loop quality and a measured model + KV footprint below 24 GB.

## Measured attempt -- Laguna S 2.1 on M1 Max (2026-08-15)

| Item | Result |
|------|--------|
| Model | `unsloth/Laguna-S-2.1-GGUF`, snapshot `5022ec3`, `UD-IQ3_XXS` |
| Weight file | 44,282,842,016 bytes (41.24 GiB) |
| Runtime | Poolside `laguna` llama.cpp branch, `04b2b72cb` / build 10008; OpenCode 1.17.18 |
| Launch | full Metal (`-ngl 99`), 16,384 context, one slot, Jinja; no speculative decoding |
| Load | 42 s; process RSS snapshot 21.7 GiB (RSS is not total unified-memory allocation) |
| Gate 1 | **pass** -- native write tool call created the exact requested file |
| Plan / JSON | **pass first try** -- six steps; `bootstrap emit plan` accepted the envelope |
| Step 1 | **pass first try** -- crate initialized; valid result and review envelopes; Continue |
| Step 2 | **fail / stopped** -- repeated long tool trajectory, filled the 16k slot once, and made no filesystem progress after writing `src/main.rs` |
| Observed server speed | about 27-29 t/s decode; about 230-260 t/s on the large prefills |
| Wall-clock | **13m02.6s**, manually stopped while still in step 2 of 6 |
| Independent `cargo test` | **fail** -- `E0599`: `.range(1..)` is unavailable on the generated clap parser; no tests were added |

This is a measured negative result. Laguna can use tools and satisfy the
harness's structured plan protocol, but this IQ3_XXS configuration did not make
bounded progress through a small coding task. Raw inference was usable; excessive
generation/tool turns and failure to test the edit dominated wall-clock. It must
not be included in README's successful-loop ranking.

## Measured attempt -- Nemotron 3.5 Lightning on M1 Max (2026-08-15)

The requested model is the Mamba-heavy hybrid: its 52 backbone blocks comprise
23 Mamba, 23 MoE, and only 6 attention blocks.

| Item | Result |
|------|--------|
| Model | `ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`, snapshot `9d425fe`, `Q4_0` |
| Weight file | 18,898,091,584 bytes (17.60 GiB); separate 1.16 GB MTP file not loaded |
| Runtime | llama.cpp build 10008; OpenCode 1.17.18 |
| Launch | full Metal, 32,768 context, one slot, q8_0 KV cache, flash attention, Jinja |
| Load | about 9.2 s; process RSS snapshot 18.3 GiB (not total unified-memory allocation) |
| Gate 1 | **pass** -- native write tool call created the exact requested file |
| Observed server speed | about 56-62 t/s decode; about 650-675 t/s on large prefills |
| Plan / JSON | **fail** -- all three attempts inspected unrelated or invalid filesystem paths instead of emitting a plan |
| Wall-clock | **1m30.43s** to terminal planner failure |
| Full loop / `cargo test` | not reached |

An Unsloth BF16-derived `UD-Q4_K_XL` snapshot (`f2d3fe3`) was also tried, but
llama.cpp builds 9770 and 10008 rejected it: expected 417 tensors, found 408.
The official ggml-org packaging separates the 18.90 GB target weights from a
1.16 GB MTP file; this run tested the base model without MTP.

This is a measured negative at a different layer from Laguna. The official
GGUF fits comfortably, loads, and performs native tool calls quickly, but is
not a drop-in planner for this harness. A narrowly scoped execution or routing
role remains plausible; a planner retry should first change its prompt/tool
policy rather than its quantization.

## Standard run record

Copy this block into `performance-analysis.md` only after a run completes:

```text
Model / source revision:
Quant / file size:
Runtime / revision:
Launch flags (context, KV, flash attention, speculation):
Resident memory:
Gate 1 -- native tool call:
Gate 2 -- strict JSON / emit recoveries:
Prefill / decode:
Loop wall-clock / steps / retries:
cargo test:
Artifact directory:
```

An unsuccessful attempt is still useful when the failure is attributable:
record unsupported architecture/format, OOM and measured allocation, malformed
tool calls, JSON failure, or timeout. Do not put a candidate in README's measured
ranking until the full loop produces independently verified working code.
