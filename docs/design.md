# Design

## CLI Argument Schema
```
--model <MODEL>    Model identifier (default: "Qwable-v1 IQ4_XS")
--iterations <N>   Number of loop iterations (default: 1)
--verbose          Enable verbose output
```

## Default Configuration
- Model: `Qwable-v1 IQ4_XS`
- Iterations: 1
- Verbosity: false

## Loop Algorithm
1. Parse CLI arguments into a `Config` struct.
2. If `iterations` is 0 or less, exit early.
3. For each iteration `i` from 1 to `iterations`:
   a. Log iteration start.
   b. Construct the command: `opencode --model <model> ...`
   c. Execute the command via `std::process::Command`.
   d. Capture stdout/stderr.
   e. Log output or errors.
   f. Continue to next iteration regardless of exit code (non-fatal errors).

## Configuration
- Model is passed as an argument to the `opencode` command.
- Iteration count controls the loop.
- Verbose mode adds per-iteration logging.

## Error Handling
- Subprocess failures (non-zero exit code) are logged but do not abort the loop.
- Fatal errors (e.g., `opencode` not found) cause immediate exit with non-zero status.
- Input validation for CLI arguments (e.g., iterations must be positive).

## Data Structures
```rust
struct Config {
    model: String,
    iterations: usize,
    verbose: bool,
}

fn run_loop(config: &Config) -> Result<()> {
    for i in 1..=config.iterations {
        // invoke opencode
    }
    Ok(())
}
```
