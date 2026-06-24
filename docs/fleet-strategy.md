# Fleet strategy — heterogeneous nodes for overnight batch agent runs

How to deploy this project across a mix of machines for **unattended, overnight,
batch-style** work: a fleet of independent systems, each handed a goal at night,
results reviewed in the morning. This is the deployment companion to the
throughput analysis — see [performance-analysis.md](performance-analysis.md)
(measured RTX 3090 baseline + M1 Max / P40 predictions) and
[older-hardware.md](older-hardware.md) (Tesla M40 / K80).

> ⚠️ **Mostly predictions.** Only **M1 Max** and **RTX 3090** have actually run
> the project. Everything about the P40, M40, K80, M10, and CPU-only nodes is
> derived from hardware characteristics and the measured 3090 baseline. Treat the
> tiers and numbers as a starting hypothesis — **measure each node once and route
> on real data** (see "Measure first").

## What the workload shape changes

Batch-by-morning is a different optimization target than interactive use:

- **Throughput-by-morning, not per-turn latency.** No human waits on any single
  `opencode` turn, so slow nodes are fine *if they finish useful work by morning*.
  The metric is "features completed per node per night."
- **Prefill still counts** (even unattended): every agent turn re-ingests a large
  prompt (system + tool schemas + file contents), so slow prompt processing
  lowers nightly step count. It's a throughput concern, not a responsiveness one.
- **Unattended stability is paramount.** 8–10 h with no operator means thermal
  headroom (passive datacenter cards need forced airflow), driver stability, and
  ECC all matter more than peak speed.
- **Power and $ are first-class.** Owned hardware is "free" only at the capital
  line; overnight you pay for **electricity**, including idle draw. The real
  metric is tokens/joule and **$/feature**, not tokens/sec.
- **Nodes are independent** (data-parallel across goals), so a heterogeneous fleet
  works — but every distinct GPU generation is a separate driver/CUDA/llama.cpp
  build to keep alive. Standardizing the stack has real value at fleet scale.

## The axes that actually separate hardware

1. **Memory bandwidth → decode speed.** Single-stream generation is
   bandwidth-bound. (Anchor: RTX 3090 at 936 GB/s does ~158 tok/s measured.)
2. **Matmul acceleration → prefill speed.** Tensor cores (Ampere/Ada), DP4A INT8
   (Pascal+), or **Intel AMX** (Sapphire Rapids+ CPUs) make prompt processing
   fast. Without any of them (Maxwell/Kepler GPUs, pre-AMX CPUs), prefill falls
   back to FP32 and collapses.
3. **Fit.** Can one device hold the model (or a usable quant)? A unified 24 GB pool
   (P40/M40) holds the 18.9 GB IQ4_XS; split-die cards (K80 2×12, M10 4×8) cannot
   and must shard across dies over a slow on-board PCIe switch; small modern cards
   need a smaller quant.
4. **Software currency.** Current CUDA 12 + driver (Ampere/Ada/Pascal/Maxwell) vs
   removed-from-CUDA-12 (Kepler → legacy CUDA 11 + EOL R470 driver).
5. **Power: load *and* idle.** Old multi-socket boxes idle at 100–200 W; Apple
   Silicon idles in single digits. Nodes that finish early and idle still cost.

## Capability tiers (efficiency- and capability-ranked)

| Tier | Hardware | Why here |
|------|----------|----------|
| **0 — efficiency king** | **M1 Max (Apple Silicon)** | tens of watts, unified memory holds full 128 k context; best tokens/joule by far. Run it first. |
| **1 — fast + modern stack** | **RTX 3090 / 4090**; **newer ≤16 GB GPUs** (Ada/Ampere) with a *fitted* quant; **Sapphire-Rapids+ Xeons (AMX)** | tensor cores / DP4A / AMX → fast prefill *and* decode on one current software stack. A newer 16 GB card with an IQ3/Q3 quant beats any old 24 GB Tesla, at lower watts. |
| **2 — old but usable 24 GB** | **Tesla P40** (good), **Tesla M40** (slow prefill) | unified 24 GB holds full IQ4_XS; P40 has DP4A, M40 doesn't. Cheap extra workers. |
| **3 — CPU-only big-RAM** | **dual/quad Xeon, many channels, AVX-512** | no GPU hassle, fits anything; bandwidth- and (pre-AMX) prefill-limited but steady. Often **beats a weak GPU** (see below). |
| **4 — avoid for a standing fleet** | **Tesla K80** (split 2×12, dead CUDA), **Tesla M10** (4×8, VDI card) | split memory + no matmul accel + worst bandwidth/software. The K80 needs a frozen legacy OS; the M10 is worse than the CPU it plugs into. |

> **VRAM capacity is a trap.** A newer 16 GB card (Tier 1) outperforms a 24 GB
> Kepler/Maxwell Tesla (Tier 2/4) on both speed and tokens/watt. Rank by
> bandwidth + matmul accel + fit, not by GB.

## The CPU floor: when a GPU is *worse* than no GPU

A GPU helps only if **both** hold: (a) the model — or a usable quant — fits in one
device's VRAM, and (b) that device's bandwidth beats your CPU's *effective*
bandwidth. Two fleet cards fail this against a strong CPU host:

- **Tesla M10 (4× GM107, 8 GB each, ~83 GB/s/die, no DP4A, 225 W).** The 18.9 GB
  model doesn't fit a single 8 GB die, so it shards across 3–4 dies; batch-1
  decode then flows *sequentially* through them, paying the **lowest bandwidth in
  the fleet plus multiple PCIe hops per token** — worse than one die. Predicted
  ~3–8 tok/s decode. A **dual 16-core Xeon (32c/64t, 384 GB)** runs CPU-only at a
  predicted ~20–40 tok/s (3B-active MoE, NUMA-aware) and fits everything. **The
  M10 burns 225 W to run slower than the CPU it's plugged into — leave it out.**
  Its only real use is its design purpose: 4 *independent small* models (≤5 GB,
  one per die) for trivial parallel goals.
- **Tesla K80** similarly half-fails fit (2×12 GB split). It at least has more
  bandwidth than the M10, but see [older-hardware.md](older-hardware.md).

**CPU nodes are a legitimate tier, not a fallback.** A 3B-active MoE only reads
~1.6 GB/token, so CPU decode is far from hopeless. Caveats:

- **Capacity ≠ bandwidth.** "384 GB of RAM" lets you *fit* big models/contexts or
  several instances; tokens/sec is set by **memory bandwidth and channel
  population**. Fill *all* channels; an under-populated board throws away
  bandwidth.
- **NUMA.** Dual/quad-socket aggregate bandwidth is only realized with NUMA-aware
  placement (`numactl`, `--numa distribute`), or by running one model instance
  per socket. Naive runs leave most of it on the floor.
- **AMX is the multiplier.** Sapphire-Rapids+ CPUs have AMX INT8/BF16 matrix
  units that llama.cpp uses — they give CPU prefill a tensor-core-like boost and
  promote a CPU host from Tier 3 to Tier 1. Pre-AMX (Skylake/Cascade AVX-512) is
  decent on decode but slow on prefill.

So **a fast Xeon + populated RAM can outrun a slow Xeon + K80/M10** — and keeps a
clean, modern software stack.

## OS choice for the oldest nodes

If you do press a K80 (Kepler) into service, **don't fight Arch's rolling release**
— building legacy CUDA 11 + the EOL R470 driver against a bleeding-edge kernel is
the slow, brittle path. Dedicate the box to **Debian stable / Ubuntu LTS** with
packaged legacy driver + CUDA against a supported kernel. OS choice makes the K80
*maintainable*; it does not make it *fast* (still split memory, no DP4A).

## Scheduling: route by prefill load, model, deadline

"Complex on better, simple on weaker" is the right shape; sharpen it:

- **Route by context / prompt size, not vague difficulty.** Old GPUs and pre-AMX
  CPUs collapse specifically on *prefill*. Send **short-context, few-file** goals
  to weak nodes; **large-context, many-file** goals to 3090 / P40 / AMX-Xeon /
  M1 Max.
- **Route the model too.** Hard goals → best node, full IQ4_XS (or a larger
  model); easy goals → weak node, smaller quant. Heterogeneous *models* matched to
  node capability.
- **Schedule for the morning deadline.** The binding constraint is "longest job
  finishes before you wake." Put longest-pole goals on the fastest node first;
  efficiency-fill with the M1 Max; hand leftover easy work to old cards only if
  their power cost earns its keep.
- **Protect against overnight failure.** Weak/old nodes are likeliest to die mid-
  run (thermal, legacy drivers). Don't place the goal you care most about *solely*
  on a K80/M10 — checkpoint, or duplicate critical goals onto a reliable node.
- **Power-gate finished nodes.** Suspend or hand more work to nodes that finish
  early so they aren't idling at 100–200 W until morning.

## Power & cost accounting

- **Use $/feature, not tokens/sec.** `$/feature ≈ (avg_watts × hours_per_feature ×
  $/kWh) / 1`. The M1 Max wins on watts; the 3090 wins on hours; old Teslas lose
  on *both* (old process node + slow), giving them the **worst marginal $/feature
  despite "free" hardware**.
- **Idle draw is a silent tax** — budget it for always-on boxes.
- **Sometimes the right call is to not run a node at all.** If a K80/M10 node's
  electricity exceeds the value of the (slow, simple) work it does versus queuing
  that work onto a fast node afterward — or a cloud spot instance — switch it off.

## A dispatcher sketch

The "give each node a goal at night" model wants a small **queue + capability-
matched dispatcher**:

- Each node advertises tags: `perf_tier`, `vram_gb`, `unified|split`, `has_amx`,
  `max_ctx`, `model_options`, `idle_watts`, `reliability`.
- Each goal carries requirements/estimates: `est_context`, `difficulty`,
  `min_model`, `deadline`.
- The scheduler matches goals to nodes (prefill-load + model fit + deadline +
  reliability), dispatches over SSH, collects `.bootstrap/` artifacts in the
  morning, and re-queues failures onto a healthier node.

Start dead simple (a tagged job file + a shell/Python loop per node); add
matching rules only once per-node measurements justify them.

## Measure first

Before trusting any tier above, measure each node **once** — throughput *and*
power — then route on the real numbers:

```bash
# throughput (toggle -fa: off for pre-Pascal, try both on Pascal+):
HF_HUB_OFFLINE=1 llama-bench -m /path/to/Qwable-v1.IQ4_XS.gguf \
  -ngl 99 -fa 0 -ctk q8_0 -ctv q8_0 -p 512 -n 128

# power: nvidia-smi -q -d POWER  (GPU)  |  IPMI / a wall meter (whole node)
```

Compute each node's **tokens/joule** and **minutes/feature**, and schedule by
those — not by the predictions in this doc.
