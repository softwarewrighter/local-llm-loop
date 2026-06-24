# Usage

## Overview

The `bootstrap` binary invokes the `opencode` tool in a configurable loop.
Each iteration runs `opencode` with the specified model and (once implemented)
a user-supplied task/prompt, then returns the output.

## Prerequisites

- Rust toolchain (installed via [rustup](https://rustup.rs/))
- The `opencode` CLI available on your `$PATH`

## Building

```bash
cargo build --release
```

The resulting binary is `target/release/bootstrap`.

## Running

Basic invocation (1 iteration, default model):

```bash
cargo run
```

Specify model, number of iterations, and verbose logging:

```bash
cargo run -- --model "Qwable-v1 IQ4_XS" --iterations 3 --verbose
```

Or using the compiled binary:

```bash
./target/release/bootstrap --model "Qwable-v1 IQ4_XS" --iterations 3 --verbose
```

## CLI Options

| Option       | Description                                 | Default               |
|--------------|---------------------------------------------|-----------------------|
| `--model`    | Model string passed to `opencode`           | `Qwable-v1 IQ4_XS`    |
| `--iterations` | Number of times to call `opencode`        | `1`                   |
| `--verbose`  | Print iteration info and exit codes to stderr | (not set)          |

## Configuring Tasks / Queries

### Current behavior

At present the program only passes `--model` to `opencode`. Because `opencode`
without a task enters interactive mode, the loop will hang after the first
iteration.

### Planned behavior

A constant list of sample tasks will be added to `src/main.rs`. The `run_loop`
function will be updated to iterate over that list and pass each task via
`--task` (or `--prompt`) so `opencode` runs in one-shot mode and returns.

### How to change the tasks (once implemented)

1. Open `src/main.rs`.
2. Locate the `TASKS` constant (a `&[&str]` array).
3. Edit the array entries to your desired prompts, e.g.:

   ```rust
   const TASKS: &[&str] = &[
       "Analyze directory layout",
       "Summarize main features",
       "Generate a README",
   ];
   ```

4. Rebuild:

   ```bash
   cargo build --release
   ```

5. Run with `--iterations` matching the number of tasks (or fewer if you want
   a subset):

   ```bash
   ./target/release/bootstrap --iterations 3 --verbose
   ```

## Troubleshooting

- **Program hangs after first iteration:** You are running the current version
  before the task list has been added. The loop still calls `opencode` without
  `--task`, causing interactive mode. Wait for the code update or manually add
  a task as described above.
- **`opencode` not found:** Ensure `opencode` is installed and its directory is
  on your `$PATH`.
- **Build errors:** Verify you have a recent Rust toolchain (`rustc --version`)
  and that the project is in the `bootstrap` directory.
