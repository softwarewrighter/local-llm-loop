# Architecture

## System Overview
The project is a Rust binary that invokes the `opencode` CLI in a configurable loop. It provides argument-driven control over model selection, iteration count, and execution behavior.

## Core Components
1. **CLI Entry Point** – Parses command-line arguments (model, iterations, etc.) using `clap`.
2. **Configuration** – Holds defaults (model defaults to `Qwable-v1 IQ4_XS`) and any user overrides.
3. **Loop Controller** – Manages iteration logic, invoking the opencode runner per cycle.
4. **Opencode Runner** – Spawns the `opencode` process with appropriate arguments, captures stdout/stderr, and handles errors.

## Data Flow
```
CLI args ──► ArgumentParser ──► Config ──► LoopController ──► OpencodeRunner ──► Output
```

- CLI args are parsed into a `Config` struct.
- The `LoopController` iterates `iterations` times, each time calling `OpencodeRunner`.
- The runner constructs a `Command` to execute `opencode --model <model> ...`.
- Output is logged or returned to the caller.

## Key Design Decisions
- **Synchronous execution**: Simple loop with blocking `Command::output()`; sufficient for typical use cases.
- **Configurable defaults**: Model defaults to `Qwable-v1 IQ4_XS`; can be overridden via `--model`.
- **Error handling**: Each iteration captures exit code; non-zero triggers a warning and continues (configurable retry policy in future).

## Future Extensions
- Config file support (YAML/TOML) for persistent settings.
- Async loop with `tokio` for concurrent invocations.
- Structured logging via `tracing`.
