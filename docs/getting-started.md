# Getting Started

Recommended: use `uv` for installation, local development, and running checks.

## Install from PyPI

```bash
uv add regelum
```

## Work on the repository

Install project dependencies from the repository root:

```bash
uv sync --all-groups
```

Install the git hooks:

```bash
uv run prek install --hook-type pre-commit --hook-type pre-push
```

Run the full local quality gate:

```bash
uv run prek run --all-files
```

Run the pendulum example:

```bash
uv run regelum-pendulum
```

Run tests:

```bash
uv run pytest
```

Run type checking:

```bash
uv run ty check src tests
```

Start the documentation server:

```bash
uv run --group docs mkdocs serve
```

Build the documentation in strict mode:

```bash
uv run --group docs mkdocs build --strict
```
