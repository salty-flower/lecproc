# Repo Working Guide

## Scope
These instructions apply to the entire repository unless a subdirectory provides a more specific `AGENTS.md`.

## Development Workflow
- Prefer running tasks through `just` recipes. Use `just format` before committing and `just check` for the full validation suite.
- Use `uv run --directory src` when invoking modules or scripts to ensure dependencies are resolved consistently.
- Avoid introducing new dependency managers; keep tooling aligned with the existing `uv` + `just` setup.

## Code Style
- Follow the existing formatting conventions enforced by `just format` and the project linters; do not hand-format unless necessary.
- Keep functions small and focused, and favour descriptive names for modules, classes, and variables.
- When editing documentation, keep the tone concise and instructional.

## Git Hygiene
- Commit logical units of work with clear messages.
- Update or add tests when changing functionality, and document notable behaviour changes in the relevant README sections if needed.

## Pull Requests
- Provide a summary of key changes and list any tests executed.
- Mention required environment variables or configuration updates if applicable.
