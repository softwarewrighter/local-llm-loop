# Measured results — RTX 3060 12 GB (this box), 2026-06-28

First **measured** wall-clock loop results on a 12 GB RTX 3060 (the README's 3060
row was "throughput only — loop pending" before this). Executes
[plan-rtx3060-12.md](plan-rtx3060-12.md). Every number below is from this box —
nothing estimated.

- **Host:** Arch Linux · **RTX 3060 12 GB** (Ampere, sm_86) · 48 threads · 251 GB RAM
- **llama.cpp:** built from source for sm_86 (CUDA 13.1), commit `dbdaece`,
  `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`
- **opencode:** 1.17.11 · harness: `bootstrap` (this repo)
- **Spec:** `examples/spec-greeter.txt` (the standard greeter CLI)
- **Loop:** `MODEL=llamacpp/<alias> ./scripts/demo-orchestrate.sh`, then an
  **independent `cargo test`** on the produced crate
- **Models / build / HF cache** all on `/data2` (root disk untouched)

## Results — **4 / 4 models produce a working, tested crate**

Ranked by measured loop wall-clock:

| # | Model | Quant | Placement | VRAM | Gate 1 | **Loop wall-clock** | Steps | Retries | `cargo test` |
|---|-------|-------|-----------|-----:|:------:|--------------------:|:-----:|:-------:|:------------:|
| 🥇 | **Ornith-1.0-9B + MTP** | Q6_K | resident | 7.7 GB | ✅ | **4m29s** | 5 (clean) | 0 | ✅ 2/2 |
| 🥈 | **Qwen3-Coder-30B-A3B** | Q4_K_M | `--n-cpu-moe 24` | 11.2 GB | ✅ | **7m20s** | 4 (clean) | 0 | ✅ 2/2 |
| 🥉 | **Ornith-1.0-9B** | Q6_K | resident | 7.5 GB | ✅ | **10m58s** | 7 (1 insert) | 0 | ✅ 2/2 |
| 4 | **Qwen3.6-35B-A3B** (no MTP) | Q4_K_M | `--n-cpu-moe 24` | 10.4 GB | ✅ | **13m40s** | 3 | 3 | ✅ 2/2 |

## `llama-bench` throughput (this box)

`-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 128`, spec-decode off:

| Model | GGUF | `--n-cpu-moe` | Prefill `pp512` | Decode `tg128` |
|-------|-----:|:-------------:|----------------:|---------------:|
| Ornith-1.0-9B Q6_K | 7.16 GiB | — (resident) | **1,580 t/s** | **41.2 t/s** |
| Qwen3-Coder-30B-A3B Q4_K_M | 17.28 GiB | 24 | 395 t/s | 30.0 t/s |
| Qwen3.6-35B-A3B Q4_K_M | 20.60 GiB | 24 | 336 t/s | 35.2 t/s |

MTP decode isn't a `llama-bench` figure (it needs the serving path); measured from
the live server: **~50–77 t/s effective with the head on vs 41 t/s baseline**
(**~1.3–1.7×**), draft acceptance **0.73–0.85**, mean accepted run ~3.2 tokens.

## Findings

1. **All four produce correct code.** Every model cleared gate 1 (native
   `tool_calls`) and emitted a crate that builds + passes `cargo test` 2/2. The
   harness's `emit` self-heal + ICRL retry loop carried the wobbles.

2. **The MTP head is the single biggest speed lever — and it's real.** The
   bundled-head GGUF (`protoLabsAI/Ornith-1.0-9B-MTP`) via `--spec-type draft-mtp`
   gives ~1.3–1.7× on decode (server-measured). Caveat: the 4m29s vs 10m58s
   wall-clock gap over plain Ornith-9B **overstates** MTP's share — the MTP run
   also drew a shorter 5-step plan vs the plain run's 7 (one inserted edition-fix).
   The robust MTP signal is the decode t/s, not the wall-clock delta.

3. **Offload is *not* the >10-min problem we hedged against — on this box.** The
   plan's fallback tier assumed offloaded MoEs would be slow; in practice
   **Qwen3-Coder-30B offloaded (7m20s) beat the resident dense Ornith-9B
   (10m58s)**. An A3B MoE reads only ~3B active params per token, so it decodes
   fast even with experts in 251 GB of RAM, while the dense 9B pays full freight
   every token. The 3060's big, fast system RAM makes offload cheap here.

4. **The 3060 is ~2.3× slower than the 5060 Ti** on the same model (Ornith-9B
   10m58s vs 4m42s) — wall-clock is the binding constraint on this card, so the
   MTP head and the fast-decoding A3B MoEs matter more here than on faster boxes.

5. **Qwen3.6-35B is the slowest (13m40s) despite a fast A3B core** — it's a
   reasoning model (`<think>`), so it generates many more tokens, and **MTP could
   not rescue it**: the unsloth `Qwen3.6-35B-A3B-UD-Q4_K_M` GGUF **does not contain
   MTP layers** (`context type MTP requested but model doesn't contain MTP layers`).
   A community MTP graft (à la the Ornith 9B) would be needed to get the head here.

## Config traps hit + fixed

- **Bundled-MTP double-load OOM.** `--spec-type draft-mtp` must be passed *alone*
  for a bundled-head GGUF. Passing `--spec-draft-model <same file>` loads a second
  full copy as the draft → OOM on 12 GB. Fixed in `start-ornith-9b-mtp.sh` and
  `start-qwen36-mtp.sh`.
- **Not every "MTP" GGUF has the head.** Verify with a load test
  (`draft-mtp` fails fast with "model doesn't contain MTP layers") before assuming
  self-spec is available.
- **Qwen3-Coder gate 1** needs `--chat-template-file Qwen3-Coder.jinja` (the
  embedded template leaks tool calls as text); confirmed working here.
- **`--n-cpu-moe` tuning (12 GB, ctx 32768, q8_0 KV):** Qwen3-Coder Q4 (17.3 GB)
  → N=24 = 11.2 GB; Qwen3.6 Q4 (20.6 GB) → N=24 = 10.4 GB, **N=20 OOMs**. Start at
  N=24 on this card.

## Reproduce

```bash
# build (one-time): CUDA sm_86 llama.cpp at /data2/llm/llama.cpp
export LD_LIBRARY_PATH=/data2/llm/llama.cpp/build/bin:/opt/cuda/lib64
# fastest working model:
LLAMA_SERVER=/data2/llm/llama.cpp/build/bin/llama-server \
MODEL=/data2/llm/models/ornith-9b-mtp/ornith-9b-mtp-kl-Q6_K.gguf \
  docs/config/arch-nvidia-3060/start-ornith-9b-mtp.sh        # port 8081, MTP on
MODEL=llamacpp/ornith-9b-mtp ./scripts/demo-orchestrate.sh   # opencode.json baseURL -> :8081
```
