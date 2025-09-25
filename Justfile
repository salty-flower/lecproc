set shell := ["zsh", "-c"]
unexport VIRTUAL_ENV

format-python:
    uv run ruff check --select I --fix
    uv run ruff format

format-sh:
    shfmt -i 2 -w scripts/*.*sh || true

format-json:
    pnpm dlx @biomejs/biome format --write --vcs-enabled true --vcs-client-kind git --vcs-use-ignore-file true --indent-style space --indent-width 2 **/*.json

[parallel]
format: format-python format-sh format-json

check-python-mypy: format-python
    uv run mypy src

check-python-basedpyright: format-python
    uv run basedpyright src

check-python-ruff: format-python
    uv run ruff check src

[parallel]
check-python: check-python-mypy check-python-basedpyright check-python-ruff

[parallel]
check: check-python

clean-pycache:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

clean-dist:
    rm -rf dist 2>/dev/null || true

clean-logs:
    rm -rf logs 2>/dev/null || true

sync-nb direction:
    uv run jupytext --sync {{if direction == "py2nb" { "analysis/*.py" } else { "analysis/*.ipynb" } }}