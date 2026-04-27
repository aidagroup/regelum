from __future__ import annotations

from pathlib import Path


_ROOT_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples"

# Keep ``regelum.examples`` importable while the actual example modules live in
# the repository-root ``examples/`` package.
__path__ = [str(_ROOT_EXAMPLES_DIR)]
