# Plan

## Implementation Steps
1. **Set up Rust project**: Initialize `cargo init`, add dependencies (`clap`, `anyhow`, `thiserror`).
2. **CLI argument parsing**: Implement `clap` derive structs for model, iterations, and verbose flag.
3. **Configuration**: Define `Config` struct and default values (model: `Qwable-v1 IQ4_XS`).
4. **Loop controller**: Implement the loop logic that invokes `opencode` per iteration.
5. **Opencode runner**: Create a function that constructs and executes the `opencode` command.
6. **Error handling**: Add error handling for subprocess failures and input validation.
7. **Testing**: Write unit tests for CLI parsing and loop logic.
8. **Documentation**: Complete this docs directory.

## Milestones
- **MVP**: Basic loop with default model, fixed iterations.
- **Configurability**: CLI args for model and iterations.
- **Polish**: Error handling, verbose mode, documentation.

## Timeline
- Week 1: Project setup, CLI parsing, basic loop.
- Week 2: Opencode runner, error handling, testing.
- Week 3: Polish, documentation, release.

## Risks & Mitigations
- **opencode CLI changes**: Wrap subprocess invocation in a separate module to allow easy updates.
- **Performance**: Use efficient I/O; avoid unnecessary allocations in the loop.
- **User confusion**: Provide clear help text and defaults.
