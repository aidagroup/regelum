# Getting Started

Install project dependencies with `uv` from the package directory:

```bash
uv sync --group dev --group docs
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
uv run pyright src tests/typecheck_api.py
```

Start the documentation server:

```bash
uv run --group docs mkdocs serve
```
