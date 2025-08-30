set shell := ["zsh", "-c"]
unexport VIRTUAL_ENV

format:
    uv run ruff check --select I --fix
    uv run ruff format

    shfmt -i 2 -w scripts/*.*sh || true

    pnpm dlx @biomejs/biome format --write --vcs-enabled true --vcs-client-kind git --vcs-use-ignore-file true --indent-style space --indent-width 2 **/*.json

check:
    uv run mypy src
    uv run basedpyright src
    uv run ruff check src

clean-pycache:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

clean-dist:
    rm -rf dist 2>/dev/null || true

clean-logs:
    rm -rf logs 2>/dev/null || true

sync-nb direction:
    uv run jupytext --sync {{if direction == "py2nb" { "analysis/*.py" } else { "analysis/*.ipynb" } }}