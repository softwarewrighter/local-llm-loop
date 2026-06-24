# Older / budget 24 GB GPUs — for an overnight batch fleet

Predicted viability of older 24 GB NVIDIA datacenter cards (Tesla **M40**,
Maxwell; Tesla **K80**, Kepler) for running this project, alongside the
already-analyzed RTX 3090, M1 Max, and P40. For the measured 3090 baseline and
the P40 prediction see [performance-analysis.md](performance-analysis.md); this
doc focuses on the *oldest* candidates and on a specific deployment shape.

> ⚠️ **All untested predictions.** The project has only been run on **M1 Max** and
> **RTX 3090**. Numbers for the P40, M40, and K80 are derived from hardware
> characteristics and the measured 3090 baseline — no figures have been observed
> on those cards. Treat as hypotheses to verify with `llama-bench`.

## The use case: overnight batch, not interactive

The target deployment is a **fleet of independent systems, each with one 24 GB
GPU**, working **in parallel on separate goals overnight** — a goal is handed to
each node at night and the results are reviewed in the morning. This reframes the
hardware question:

- **Throughput-by-morning matters; per-turn latency does not.** No human is
  waiting on any single `opencode` turn, so a slow card is acceptable *as long as
  it finishes useful work by morning*. The metric is "features completed per node
  per night," not tokens/sec felt interactively.
- **Prefill still counts** — even unattended. Every agent turn re-ingests a large
  prompt (system + tool schemas + file contents), so slow prompt processing
  directly reduces how many steps a node completes overnight. It just isn't a
  *responsiveness* concern.
- **Unattended stability is paramount.** An 8–10 h run with no operator means
  thermal headroom (these are passively-cooled cards needing forced airflow),
  driver stability, and ECC memory (all three Teslas have it) matter more than
  peak speed.
- **Nodes are independent (data-parallel across goals),** so a heterogeneous fleet
  is fine in principle — but every distinct GPU generation is a separate
  driver/CUDA/llama.cpp build to maintain. Standardizing the stack has real value.

## Three axes decide it

1. **DP4A INT8 (Pascal+).** llama.cpp's fast quantized matmul (MMQ) rides Pascal's
   DP4A instruction. **P40 has it; M40 (Maxwell) and K80 (Kepler) do not** — they
   fall back to dequant → FP32/FP16, crippling prefill. None of these cards have
   tensor cores.
2. **Unified vs split memory.** P40 and M40 are a single 24 GB pool that holds the
   18.9 GB model natively. **The K80 is two 12 GB dies** — the model must be
   tensor-split across both, with cross-die traffic over an on-board PCIe switch.
3. **Software currency.** P40 (CC 6.1) and M40 (CC 5.2) are still in current CUDA
   12.x (Maxwell deprecated but present). **Kepler (CC 3.7) was removed in CUDA
   12** — the K80 needs legacy CUDA 11 + the EOL R470 driver, which conflicts with
   a modern Arch kernel and any newer card on the same host.

## Comparison

| | RTX 3090 | M1 Max | Tesla P40 | Tesla M40 | Tesla K80 |
|---|---|---|---|---|---|
| Arch / year | Ampere '20 | Apple '21 | Pascal '16 | Maxwell '15 | Kepler '14 |
| Compute capability | 8.6 | — | 6.1 | 5.2 | 3.7 |
| Memory | 24 GB | unified | 24 GB | 24 GB | **2 × 12 GB** |
| Bandwidth | 936 GB/s | ~400 | 346 | 288 | 240 /die |
| DP4A INT8 (MMQ) | tensor | — | ~47 TOPS | **none** | **none** |
| Tensor cores | yes | no | no | no | no |
| FP32 | ~35 TFLOPS | ~10 | ~12 | ~6.8 | ~4.4 /die |
| Modern CUDA 12 / driver | ✅ | n/a | ✅ | ⚠️ deprecated | ❌ removed |
| Fits 18.9 GB model on one pool | ✅ | ✅ | ✅ | ✅ | ❌ must split |

### Predicted throughput (relative to the measured 3090)

Decode scales with memory bandwidth (anchored to the 3090's measured 158 tok/s);
prefill depends on the matmul path and is far softer.

| Metric | RTX 3090 (measured) | P40 (pred.) | M40 (pred.) | K80 (pred.) |
|--------|--------------------:|------------:|------------:|------------:|
| Generation (`tg128`) | ~158 tok/s | ~45–55 | ~35–45 | ~15–30 |
| Prompt / prefill (`pp512`) | ~3,605 tok/s | ~400–700 | ~80–200 | ~50–150 |

## Tesla M40 (Maxwell, 24 GB)

Works, and is the **better of the two old cards**. Single 24 GB pool holds the
model; decode (~35–45 tok/s) is only modestly below the P40 because it just tracks
bandwidth (288 vs 346 GB/s). The weakness is **prefill**: no DP4A means quantized
matmul falls back to ~6.8 TFLOPS FP32 cuBLAS, ~3–5× slower than the P40's
DP4A path. For overnight batch that's tolerable — it lowers nightly step count but
doesn't block progress.

Gotchas:
- **Build llama.cpp with `sm_52`** (`-DCMAKE_CUDA_ARCHITECTURES=52`); stock
  binaries may target newer arches only.
- Maxwell driver support is nearing legacy — fine now, but plan for it.
- Run with `-fa 0` (flash attention is weak/absent pre-Pascal).
- A simpler k-quant (`Q4_K_M`) may run better than the i-quant IQ4_XS here, since
  there are no tensor cores to hide the heavier dequant.

## Tesla K80 (Kepler, 2 × 12 GB)

**Not recommended** — the only card here I'd actively avoid for a fleet, for
reasons that outweigh its low price:

- **It's two 12 GB dies, not a 24 GB card.** The 18.9 GB model can't live on one
  die; you must tensor-split across both, paying cross-die PCIe traffic every
  layer boundary. KV-cache placement across two pools is also awkward.
- **Dropped by CUDA 12.** Requires legacy CUDA 11 + the EOL R470 driver — which
  won't coexist cleanly with a modern Arch kernel or any newer GPU on the host.
  For a fleet you maintain unattended, this is a recurring liability, not a
  one-time setup cost.
- **Slowest on every axis** — lowest bandwidth (240 GB/s/die), no DP4A, no tensor
  cores, oldest FP32. Predicted decode ~15–30 tok/s, prefill the worst of the set.

If one already exists and you only want to *experiment*, it can technically run a
split model on an old toolchain — but it's a poor foundation for a standing
overnight fleet.

## Recommendation for the fleet

- **Standardize on the P40** among the old 24 GB cards: it's the only one with
  DP4A (usable prefill), a true unified 24 GB pool, and current CUDA/driver
  support — one software stack to maintain across nodes.
- **M40 as acceptable filler.** If you already own M40s, they're fine cheap nodes
  for overnight batch; expect noticeably lower nightly throughput than P40 nodes
  due to slow prefill. Keep the stack `sm_52`-aware.
- **Skip the K80** for a standing fleet — split memory and a dead software stack
  make it more maintenance than it's worth; a P40 is the same price class and far
  better.
- **Mixing generations is OK** since nodes run independently, but every extra
  generation adds a driver/CUDA/llama.cpp build to keep alive — a real cost at
  fleet scale. A homogeneous P40 fleet is the low-friction choice.
- Whatever the card: ensure forced airflow over these passive Teslas, leave ECC
  on, and prefer the largest stable context (~32k on the 24 GB unified cards) so
  long overnight loops don't truncate.

> If the budget allows even used **RTX 3090s**, one per node would dwarf all of
> these (≈2–3× decode, several× prefill) and keep the modern stack — worth pricing
> against a larger count of old Teslas for the same total throughput.

## Verify

Drop the card in and run the same bench used elsewhere, toggling flash attention:

```bash
HF_HUB_OFFLINE=1 llama-bench -m /path/to/Qwable-v1.IQ4_XS.gguf \
  -ngl 99 -fa 0 -ctk q8_0 -ctv q8_0 -p 512 -n 128   # try -fa 1 too on Pascal+
```

Compare the `pp512` / `tg128` rows directly against the
[3090 baseline](performance-analysis.md).
