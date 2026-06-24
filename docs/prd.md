# Product Requirements Document (PRD)

## Problem Statement
Users need a simple, configurable tool to run the `opencode` CLI in a loop, with control over the model selection, number of iterations, and execution behavior.

## Goals
- Allow users to specify the model to use (default: `Qwable-v1 IQ4_XS`).
- Support configurable iteration count.
- Provide clear feedback on each iteration's success/failure.
- Keep the tool easy to use via CLI arguments.

## User Stories
1. As a user, I want to run `opencode` with a specific model for a set number of iterations.
2. As a user, I want to override the default model via a command-line flag.
3. As a user, I want to see the output of each iteration for debugging or logging purposes.
4. As a user, I want the tool to handle errors gracefully and continue on failure.

## Functional Requirements
- Parse CLI arguments for model, iterations, and optional flags.
- Default model to `Qwable-v1 IQ4_XS` if not specified.
- Loop the specified number of times, invoking `opencode` each time.
- Capture and display stdout/stderr from each invocation.
- Exit with appropriate status code (0 for success, non-zero for fatal errors).

## Non-Functional Requirements
- **Performance**: Minimal overhead per iteration; no unnecessary allocations.
- **Reliability**: Graceful handling of subprocess failures.
- **Configurability**: All key parameters (model, iterations) should be adjustable via CLI.
- **Simplicity**: Minimal dependencies; easy to build and deploy.

## Out of Scope
- GUI interface.
- Persistent configuration files (planned for future).
- Concurrent/parallel execution (planned for future).

## Success Metrics
- Tool runs successfully with default settings for at least 10 iterations.
- Users can override the model via CLI without errors.
- Error rate per iteration is below 5% in typical usage.
