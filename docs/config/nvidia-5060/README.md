# RTX 5060 Ti 16 GB (Blackwell / sm_120) — local model configs

Start-script + download templates for testing small coding models on a **16 GB
RTX 5060 Ti** node. Method and the two gates carry over from the
[24 GB RTX 3090 model search](../../nvidia-3090-poc-summary.md); results for this
node go in [../../nvidia-5060-poc-summary.md](../../nvidia-5060-poc-summary.md).

> **Hardware note:** the card in the test box is an **RTX 5060 Ti 16 GB** (GB206
> Blackwell, sm_120, ~448 GB/s GDDR7) — the 16 GB Blackwell part. The
> [plan](../../plan-rtx5060-16.md) calls it "5060-16"; the method is identical.

```
docs/config/nvidia-5060/
├── README.md                 # this file
├── download-models.sh        # pull candidate GGUFs to /disk1/models (uv venv + hf)
├── start-gptoss-20b.sh       # gpt-oss-20b MXFP4 — FP4 showcase, fits 16 GB whole
├── start-qwen3-coder.sh      # Qwen3-Coder-30B-A3B MoE — proven loop-completer (--n-cpu-moe)
├── start-phi4.sh             # Phi-4-14B dense — fits whole, structured-output candidate
└── start-qwen3-8b.sh         # Qwen3-8B dense — solo or + Qwen3-0.6B draft (spec-decode)
```

## Build llama.cpp for Blackwell (sm_120)

An Ampere/Ada build will **not** run on Blackwell. Build a CUDA `sm_120` server:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /disk1/build/llama.cpp
cd /disk1/build/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=ON -G Ninja
cmake --build build --target llama-server llama-bench -j
build/bin/llama-server --list-devices    # must list the RTX 5060 Ti
```

The start scripts default `LLAMA_SERVER` to `/disk1/build/llama.cpp/build/bin/llama-server`
(override via env).

## Disk / cache placement

Everything heavy lives on **`/disk1`** (most free space); the small root
partition is never touched:

- models → `/disk1/models/<model>/`
- HF cache → `/disk1/hf-cache` (`HF_HOME`); also pinned in `~/.bashrc`
- llama.cpp build → `/disk1/build/llama.cpp`
- the `hf` downloader → a **uv** venv at `/disk1/venvs/llm-loop` (no system pip/conda)

## The two gates (carry over from the 3090)

A model completes the loop only if it (1) emits **native `tool_calls`** llama.cpp
parses, and (2) emits **strict JSON** for the plan/step/review envelopes. Small
models flunk gate 2 more often. See the
[5060 POC summary](../../nvidia-5060-poc-summary.md) for the per-model matrix.

## Run one

```bash
docs/config/nvidia-5060/start-gptoss-20b.sh        # serves :8080, alias gptoss-20b
# in another shell:
MODEL=llamacpp/gptoss-20b ./scripts/demo-orchestrate.sh
```

Run **one** server at a time (they all bind `:8080`); point opencode at the
matching alias in [`../opencode.json`](../opencode.json).
