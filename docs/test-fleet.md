# Test fleet — hardware inventory & per-box role

The physical machines this project is being validated on. The README's
"[Which model on which box](../README.md#which-model-on-which-box)" table groups
**measured loop results by GPU tier**; this doc is the **full box inventory** —
several boxes share a GPU type but differ in CPU/RAM/OS, and several are owned but
not yet measured.

> **Status legend:** ✅ measured (loop + `cargo test`) · 🔜 owned, to test ·
> 🔭 planned/other (much later). Fill the **CPU / RAM** blanks as each box is benched.

## Priorities (why we test what we test)

- **24 GB is the sweet spot** for local LLM coding agents — it holds a 30–35B MoE
  coder whole (fast, no offload). **Primary focus.**
- **16 GB is great if it works** — the RTX 5060 Ti already does (gpt-oss-20b 1m13s,
  Ornith-9B+MTP 2m15s). The **A2-16 GB** is the next 16 GB datapoint.
- **12 GB: low expectations for *coding*.** The RTX 3060 runs the loop but slowly
  (Ornith-9B+MTP 4m29s … Qwen3.6-35B 13m40s). The open question for 12 GB is
  whether it's better aimed at **admin / non-coding agent tasks** (log triage,
  file ops, summaries, scheduled jobs) rather than authoring code. **To explore.**
- **Apple Silicon** spans a wide range: the M1 Max (64 GB) holds anything; the
  small **M1 (8 GB)** and **M2 Pro (18 GB)** test the low end of unified memory.

## The fleet

| # | Box | GPU / accelerator | VRAM | CPU / RAM | OS | Status | Intended role |
|---|-----|-------------------|------|-----------|----|--------|---------------|
| 1 | **3060-A** | RTX 3060 | 12 GB | 48 threads · 251 GB RAM | Arch | ✅ measured (4/4 green) | 12 GB coding baseline; **admin-task candidate** |
| 2 | **3060-B** | RTX 3060 | 12 GB | _TBD (differs from 3060-A)_ | _TBD_ | 🔜 to test | confirm 12 GB result on different CPU/RAM |
| 3 | **5060 Ti** | RTX 5060 Ti (Blackwell, FP4) | 16 GB | _TBD_ · _TBD_ | Arch | ✅ measured (5 models) | 16 GB coding — **works well** |
| 4 | **A2** | NVIDIA A2 (Ampere) | 16 GB | _TBD_ | _TBD_ | 🔜 to test | 2nd 16 GB datapoint (low-power datacenter card) |
| 5 | **3090** | RTX 3090 | 24 GB | _TBD_ | Arch | ✅ measured (qwen3-coder ~3m30s) | **24 GB sweet spot — primary focus** |
| 6 | **dual-P40** | 2× Tesla P40 (Pascal) | 2× 24 GB | _TBD (older gen)_ | _TBD_ | 🔜 to test | 24 GB sweet spot on cheap older silicon (DP4A) |
| 7 | **M1 Max** | Apple M1 Max | 64 GB unified | — · 64 GB | macOS | ✅ measured | holds *any* model; MLX path |
| 8 | **M2 Pro** | Apple M2 Pro | 18 GB unified | — · 18 GB | macOS | 🔜 to test | mid unified-memory Apple Silicon |
| 9 | **M1 (base)** | Apple M1 | 8 GB unified | — · 8 GB | macOS | 🔜 to test | low end — likely admin/small-model only |

> **Count:** 9 boxes planned (5 already exercised: 3060-A, 5060 Ti, 3090, M1 Max,
> plus the in-progress set). Other systems with other GPUs exist and are deferred
> to "much later" (see [fleet-strategy.md](fleet-strategy.md) for the deployment
> view and [older-hardware.md](older-hardware.md) for older 24 GB Tesla analysis).

## What each new box is meant to answer

- **3060-B** — does the 12 GB result hold across a *different* CPU/RAM host? (3060-A
  had 251 GB RAM + 48 threads, which helps MoE expert-offload; a leaner host may
  shift the offloaded-MoE numbers.)
- **A2-16 GB** — is 16 GB coding viable on a *low-power Ampere* datacenter card
  (no Blackwell FP4 lane like the 5060), and how does its bandwidth compare?
- **dual-P40** — can the 24 GB sweet spot be reached cheaply on Pascal? (P40 has
  DP4A INT8 but no tensor cores; older `older-hardware.md` predicted it usable.)
- **M2 Pro 18 GB / M1 8 GB** — the low end of unified memory: what fits, and where
  the line is between "codes" and "admin tasks only."
