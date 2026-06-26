# Local model configuration

Reference configs for the two processes this project depends on:

1. **`llama-server`** (llama.cpp) — serves the Qwable-v1 GGUF over an
   OpenAI-compatible HTTP endpoint.
2. **`opencode`** — the coding agent `bootstrap` shells out to; it talks to
   `llama-server` as a custom provider.

```
docs/config/
├── opencode.json                      # portable — verified-working models, one per alias
├── mac/start-qwable.sh                # Apple Silicon / Metal
├── arch-nvidia/start-qwable.sh        # Arch Linux / NVIDIA CUDA (generic)
├── arch-nvidia-3060/                  # 12 GB RTX 3060 — MoE expert offload to RAM
└── nvidia-3090/                       # 24 GB RTX 3090 — one script per model:
    ├── start-qwen3-coder.sh           #   Qwen3-Coder-30B-A3B MoE (alias qwen3-coder, verified)
    ├── start-gemma4-26b.sh            #   Gemma-4-26B-A4B MoE     (alias gemma4-26b, best quality)
    ├── start-gemma4-31b.sh            #   Gemma-4-31B + E2B draft (alias gemma4-31b, dense spec-decode)
    └── start-qwable.sh                #   Qwable-v1 baseline      (alias qwable)
```

The harness is **model-agnostic**. Each model has its **own opencode alias, its own
GGUF dir, and its own start script** — point opencode at the alias of whichever
server is currently bound to `:8080` (run ONE at a time). `opencode.json` is
platform-independent (points at `localhost`, same XDG path everywhere) and lists the
verified-working models; **every model needs a zero `cost` block** or opencode 1.x's
cost-calc throws `DecimalError`. The full model-search (which models pass/fail and
why) is in [../nvidia-3090-poc-summary.md](../nvidia-3090-poc-summary.md).

## Install / placement

| File | Goes to | Notes |
|------|---------|-------|
| `opencode.json` | `~/.config/opencode/opencode.json` | Same path on macOS and Linux (XDG) |
| `start-qwable.sh` | anywhere (e.g. next to the model) | `chmod +x`; edit `MODEL` to your GGUF path |

## llama-server command & parameters

```bash
llama-server \
  -m "$MODEL" \                 # path to the GGUF model file
  --alias qwable \              # model id clients see  → opencode "qwable"
  --gpu-layers 99 \             # 99 = offload all layers to the GPU
  --ctx-size 131072 \          # total context window (tokens), split across slots
  --parallel 1 \                # number of concurrent slots
  --cont-batching \             # continuous batching (throughput)
  --flash-attn on \             # flash attention (Metal and CUDA)
  --cache-type-k q8_0 \         # quantize K cache → less KV memory
  --cache-type-v q8_0 \         # quantize V cache → less KV memory
  --jinja \                     # use the model's Jinja chat template (tool calls!)
  --host 127.0.0.1 --port 8080 \
  --n-predict 4096 \            # max tokens per generation (runaway guard)
  --api-key local
```

| Flag | Why it matters |
|------|----------------|
| `--alias` | The model id opencode targets (`llamacpp/qwable`). **Must match** the model key in `opencode.json`. |
| `--gpu-layers` | `99` offloads everything. Lower it if VRAM is tight (NVIDIA). |
| `--ctx-size` | Server's max context. **Must be ≥** opencode's `limit.context`. |
| `--n-predict` | Caps tokens per response — the guard against a runaway generation. Keep **≥** opencode's `limit.output` so opencode controls length normally. |
| `--jinja` | Applies the GGUF's chat template; required for correct **tool calling** (opencode `tools: true`). Without it, tool calls and stop tokens misbehave. |
| `--cache-type-k/v q8_0` | Halves KV-cache memory vs f16 — important for large `--ctx-size`. |
| `--api-key` | **Must match** `opencode.json` `provider.options.apiKey`. |
| `--host/--port` | **Must match** `opencode.json` `provider.options.baseURL`. |

## opencode.json settings

```jsonc
"llamacpp": {
  "npm": "@ai-sdk/openai-compatible",   // adapter for any /v1 OpenAI-style server
  "options": {
    "baseURL": "http://localhost:8080/v1",  // == llama-server host:port + /v1
    "apiKey": "local"                        // == llama-server --api-key
  },
  "models": {
    "qwable": {                           // == llama-server --alias
      "tools": true,                      // enable tool calling (needs --jinja server-side)
      "limit": { "context": 32768, "output": 4096 }
    }
  }
}
```

| Setting | Why it matters |
|---------|----------------|
| `npm` | Must be `@ai-sdk/openai-compatible` (NOT `package`). The adapter for any OpenAI-style `/v1` server. |
| model key (`qwable`) | **Must match** `--alias`. Referenced project-wide as `llamacpp/qwable`. |
| `options.baseURL` | **Must match** the server's `--host:--port` (+ `/v1`). |
| `options.apiKey` | **Must match** the server's `--api-key`. |
| `tools` | `true` enables tool use; only works if the server runs with `--jinja`. |
| `limit.output` | opencode's max requested tokens. Keep **≤** server `--n-predict`. |
| `limit.context` | opencode compacts the conversation at this size. Keep **≤** server `--ctx-size`. |

## Compatibility matrix (must-match pairs)

| llama-server | ⟷ | opencode.json |
|--------------|---|---------------|
| `--alias qwable` | = | model key `qwable` (used as `llamacpp/qwable`) |
| `--host` / `--port` (`127.0.0.1:8080`) | = | `options.baseURL` (`http://localhost:8080/v1`) |
| `--api-key local` | = | `options.apiKey` (`local`) |
| `--n-predict 4096` | ≥ | `limit.output` (`4096`) |
| `--ctx-size 131072` | ≥ | `limit.context` (`32768`) |
| `--jinja` (chat template) | required by | `tools: true` |
| exposes `/v1` (OpenAI-compatible) | required by | `npm: @ai-sdk/openai-compatible` |

If any pair is out of sync you'll see auth failures, "model not found", silently
truncated responses, broken tool calls, or context errors.

## What you need to recreate the setup

The two config files alone are **not** enough. Full checklist (same on macOS and
Arch):

1. **The model GGUF** — `Qwable-v1.IQ4_XS.gguf`. Large; not in this repo. Place it
   locally and point the script's `MODEL` at it.
2. **llama.cpp `llama-server`** on `$PATH`, built for your GPU (see below).
3. **opencode** CLI installed and on `$PATH`.
4. **Rust toolchain** (edition 2024 / `rustc` ≥ 1.85) to build `bootstrap`.
5. **`opencode.json`** copied to `~/.config/opencode/opencode.json`.

### macOS (Apple Silicon / Metal)

- llama.cpp builds with Metal by default on Apple Silicon. Install via Homebrew
  (`brew install llama.cpp`) or build from source.
- Unified memory comfortably holds the 128k-token KV cache → `mac/start-qwable.sh`
  defaults `CTX=131072`.

### Arch Linux (NVIDIA / CUDA)

- Install the NVIDIA driver + CUDA: `sudo pacman -S nvidia cuda`.
- Build llama.cpp with CUDA:
  ```bash
  cmake -B build -DGGML_CUDA=ON
  cmake --build build --config Release -j
  # llama-server → build/bin/llama-server
  ```
  (Or use a CUDA-enabled package/container if you prefer.)
- **VRAM is the real constraint.** A single consumer GPU usually can't hold the
  model *plus* a 128k KV cache, so `arch-nvidia/start-qwable.sh` defaults
  `CTX=32768` (matching opencode's `limit.context`). Raise it only with the VRAM
  to spare; drop `GPU_LAYERS` to offload fewer layers if you run out.
- Everything else — the `llama-server` flags, `opencode.json`, and the
  compatibility matrix — is identical to macOS.

> Same software, same flags, same config. The only platform differences are the
> GPU build backend (Metal vs CUDA) and the context/VRAM budget.
