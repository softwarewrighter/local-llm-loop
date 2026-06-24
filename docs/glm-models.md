# Running GLM models with this harness

`bootstrap` is model-agnostic: it shells out to `opencode run --model
<provider/model>`, and opencode talks to any OpenAI-compatible endpoint. So the
same plan → execute → review loop that drives the local Qwable model
([config](config/README.md)) can drive a much larger frontier model instead —
trading speed for quality on a big-RAM node. This doc shows how to point the
harness at **GLM-5.2**.

> ⚠️ **Untested configuration / estimated numbers.** This repo has only been run
> against Qwable on the M1 Max and RTX 3090. The GLM-5.2 setup below is a worked
> configuration, not a measured result — confirm sizes and throughput on your own
> node. See [performance-analysis.md](performance-analysis.md) and
> [fleet-strategy.md](fleet-strategy.md) for the measurement method and the node
> inventory this assumes.

## GLM-5.2 — 3-bit on a 512 GB Gold node + A2-16G

**GLM-5.2** (Z.ai / Zhipu, MIT license, June 2026) is a **~754B-parameter MoE with
~40B active parameters per token** and a 1M-token context — a frontier-class
agentic-coding model. At those sizes it is a **big-RAM, CPU-resident** job with a
GPU offloading only what fits; no single GPU you own can hold it.

### Why run it here

For **complex overnight-batch goals** where quality matters more than turn
latency, GLM-5.2 is a large step up from the local Qwable model. It's slow on this
hardware (single-digit tok/s), which is exactly why it belongs on a batch node:
hand it a goal at night, review in the morning.

### The target node

| Component | This config |
|-----------|-------------|
| CPU host | one **Gold** Xeon (5315Y, 8ch ~188 GB/s; or 6230, 6ch ~141 GB/s) — AVX-512 |
| System RAM | **512 GB** (holds the 282 GB model + KV + headroom) |
| GPU | **A2-16G** (Ampere, fanless, rack-friendly) for the dense/attention layers |
| Chassis | rack server (the A2 needs server airflow — see the chassis rule in [fleet-strategy.md](fleet-strategy.md)) |

### Sizing — pick the 3-bit dynamic quant

The relevant rows from the Unsloth GGUF repo (full list linked under Sources):

| Quant | Size | Fits 512 GB? |
|-------|------|--------------|
| UD-IQ2_M (2-bit) | 239 GB | ✅ lots of room |
| **UD-IQ3_XXS (3-bit)** | **282 GB** | ✅ **~210 GB headroom for KV + OS** — the target |
| UD-IQ4_XS (4-bit) | 365 GB | ✅ but tight once KV grows; prefer ≥640 GB for 4-bit |

**UD-IQ3_XXS (282 GB)** is the sweet spot here: it fits 512 GB with comfortable
headroom, and because Unsloth's *dynamic* quants keep important layers at higher
precision, its quality sits much closer to 4-bit than a naive 3-bit would.

### 1. Get the model (no surprise re-downloads)

```bash
# ~282 GB — fetch the UD-IQ3_XXS shards into your HF cache (HF_HOME)
huggingface-cli download unsloth/GLM-5.2-GGUF \
  --include "*UD-IQ3_XXS*" --local-dir /path/to/models/GLM-5.2
```

Ensure your `llama.cpp` build is **new enough for the GLM-5.2 architecture**
(IndexShare / glm-moe); an older build will reject the model at load.

### 2. Start llama-server with MoE expert offload

The trick is keeping the **giant expert tensors in the 512 GB RAM** and putting as
many **dense/attention layers on the A2 as its 16 GB allows** (the A2 mainly buys
you faster *prefill* — the CPU's weak spot — not faster decode):

```bash
HF_HUB_OFFLINE=1 llama-server \
  -m /path/to/models/GLM-5.2/GLM-5.2-UD-IQ3_XXS-00001-of-*.gguf \
  --alias glm-5.2 \
  -ngl 99 \                              # nominally all layers to GPU…
  -ot ".ffn_.*_exps.=CPU" \              # …but force MoE experts back to RAM
  --ctx-size 32768 \                     # modest ctx protects RAM for experts
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --flash-attn on --jinja \
  --host 127.0.0.1 --port 8080 \
  --n-predict 8192 --api-key local
```

**Tuning the A2's 16 GB:** with `-ngl 99 -ot …exps…=CPU`, the GPU holds the
dense/attention weights + KV cache. For a 754B model that dense portion may exceed
16 GB — if the server OOMs on the GPU, **lower `-ngl`** (offload fewer dense
layers) and/or keep the context small until it fits. The A2 offloading "some
layers" is expected: most of the model lives in RAM regardless.

> Run GLM-5.2 on a **different port** (or stop the Qwable server first) if a
> Qwable `llama-server` is already bound to `:8080`.

### 3. Add it to `opencode.json`

Add a model entry under the existing `llamacpp` provider (keep the alias, port,
api-key, and `limit.context` in sync with the server — see the compatibility
matrix in [config/README.md](config/README.md)):

```jsonc
"llamacpp": {
  "npm": "@ai-sdk/openai-compatible",
  "options": { "baseURL": "http://127.0.0.1:8080/v1", "apiKey": "local" },
  "models": {
    "glm-5.2": {                         // == llama-server --alias
      "name": "GLM-5.2 UD-IQ3_XXS",
      "tools": true,                     // needs server --jinja
      "limit": { "context": 32768, "output": 8192 }
    }
  }
}
```

### 4. Run the harness against it

```bash
bootstrap orchestrate path/to/spec.txt \
  --model llamacpp/glm-5.2 \
  --work-dir /path/to/target/project \
  --max-steps 30 --verbose
```

Everything else (artifacts under `.bootstrap/`, the plan/execute/review loop) is
unchanged — only the model behind the endpoint differs.

### Performance & expectations

- **Decode** is bound by RAM bandwidth: ~40B active params at 3-bit ≈ ~14–16 GB
  read per token, so a ~140–188 GB/s Gold yields a predicted **~5–9 tok/s** (the
  A2 changes this little — the weights are in RAM). Fine for overnight batch,
  painful interactively.
- **Prefill** is the CPU's weak spot (no AMX on these Golds); the A2's tensor
  cores accelerate whatever dense layers it holds, which is the main reason to
  bother attaching it.
- **Context:** GLM-5.2 supports 1M tokens, but KV cache at long context competes
  with the experts for RAM — keep `--ctx-size` modest (32–64k) for batch coding.

### Variants: CPU-only and RTX 3060-12G

The A2 is optional. The model lives in RAM either way; a GPU only accelerates the
*dense/attention* layers (mainly prefill). Two variants on the same 512 GB host:

**A) CPU-only (no GPU).** Works as long as the 282 GB quant fits RAM (it does, in
512 GB). Drop the GPU flags:

```bash
HF_HUB_OFFLINE=1 llama-server \
  -m /path/GLM-5.2-UD-IQ3_XXS-00001-of-*.gguf --alias glm-5.2 \
  -ngl 0 \                                  # pure CPU
  --ctx-size 32768 --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja -t 8 \                            # -t = physical cores (8 on a 5315Y)
  --host 127.0.0.1 --port 8080 --n-predict 8192 --api-key local
# NUMA-pin on a dual socket: numactl --membind=0 --cpunodebind=0 llama-server …
```

The **bigger CPU-only cost is prefill, not decode**: with no tensor cores (and no
AMX on these Golds), ingesting a large agent prompt takes tens of seconds to
minutes per turn. Decode stays ~5–9 tok/s. The 5315Y is bandwidth-rich (8 ch →
good decode) but core-poor (8 cores → weak prefill); the 20-core 6230 prefills
better if you have the choice. This variant is strictly overnight-batch.

**B) RTX 3060-12G (a *better* test accelerator than the A2 for raw speed).** The
3060 has ~360 GB/s and more compute than the A2's ~200 GB/s, so it speeds prefill
more — at the cost of a smaller 12 GB budget for the GPU-resident dense layers.
Same expert-offload pattern as §2, just tune `-ngl` down if the 12 GB OOMs:

```bash
HF_HUB_OFFLINE=1 llama-server -m /path/GLM-5.2-UD-IQ3_XXS-*.gguf --alias glm-5.2 \
  -ngl 99 -ot ".ffn_.*_exps.=CPU" \         # dense→GPU, experts→512 GB RAM
  --ctx-size 32768 -fa on -ctk q8_0 -ctv q8_0 --jinja \
  --host 127.0.0.1 --port 8080 --n-predict 8192 --api-key local
# if the GPU OOMs: drop -ngl (e.g. 40, then lower) until the dense layers + KV fit 12 GB
```

The 3060 is a consumer card, so this assumes a **workstation** host (chassis rule,
[fleet-strategy.md](fleet-strategy.md)). Decode is still RAM-bound (~5–9 tok/s);
prefill gets the lift. The A2's edge is later, for a standing node: ~40–60 W and
fanless/rack-friendly, not raw speed.

> **Throughput caveat vs the old Teslas.** The "M40 ≈ 35–45 / P40 ≈ 45–55 tok/s"
> figures elsewhere in these docs are for **Qwable (3B active)**. GLM-5.2 has
> **~40B active params** — ~10× more data per token — so even before the
> CPU-vs-GPU bandwidth gap, GLM-5.2 here runs **~5–10× slower in raw tok/s** than
> those Tesla-on-Qwable numbers. No Tesla can run GLM-5.2 at all (282 GB ≫ 24 GB);
> it's a frontier model on a CPU vs a 34B model on a GPU, not a like-for-like
> comparison.

### Caveats

- Single-digit tok/s makes this a **batch-only** model on this hardware; do not
  schedule latency-sensitive work on it.
- 4-bit (UD-IQ4_XS, 365 GB) is preferable for quality but wants **≥640 GB RAM**
  once KV grows; 3-bit is the practical fit for a 512 GB node.
- Verify the `llama.cpp` build supports GLM-5.2; verify the model loads before
  wiring it into a long overnight run.

## Sources

- [unsloth/GLM-5.2-GGUF · Hugging Face](https://huggingface.co/unsloth/GLM-5.2-GGUF) — quant sizes
- [GLM-5.2 — How to Run Locally | Unsloth Docs](https://unsloth.ai/docs/models/glm-5.2) — memory/offload guidance
- [zai-org/GLM-5.2 · Hugging Face](https://huggingface.co/zai-org/GLM-5.2) — model card
