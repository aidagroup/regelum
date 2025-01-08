"""Regelum: A High-Performance Node-Based Computation Framework.

Regelum is a powerful framework for building and executing computational graphs,
with support for parallel execution, symbolic computation, and automatic dependency resolution.

Key Features:
- Node-based computation with automatic dependency tracking
- Parallel execution using Dask
- Symbolic computation support via CasADi
- Flexible graph composition and manipulation
- Built-in logging and debugging tools
"""

from contextlib import contextmanager
from typing import Dict, Any, Iterator

from regelum.node.core.globals import _SYMBOLIC_INFERENCE_ACTIVE
from regelum.node.core.variable import Variable
from regelum.node.core.inputs import Inputs

__version__ = "1.0.0"
__author__ = "Regelum Team"
__license__ = "MIT"

# Core components
from regelum.node.base import Node
from regelum.node.core.types import ResolveStatus
from regelum.node.graph import Graph
from regelum.node.logging import Clock, StepCounter, Logger
from regelum.node.parallel import ParallelGraph


@contextmanager
def symbolic_mode() -> Iterator[None]:
    """Context manager for symbolic mode."""
    _SYMBOLIC_INFERENCE_ACTIVE.value = True
    try:
        yield
    finally:
        _SYMBOLIC_INFERENCE_ACTIVE.value = False


def get_version() -> str:
    """Get the version of Regelum."""
    return __version__


# Package metadata
metadata: Dict[str, Any] = {
    "name": "regelum",
    "version": __version__,
    "author": __author__,
    "license": __license__,
    "description": "High-Performance Node-Based Computation Framework",
    "requires": [
        "numpy",
        "casadi",
        "torch",
        "dask",
        "dask.distributed",
    ],
}

__all__ = [
    # Core classes
    "Node",
    "Graph",
    "ParallelGraph",
    "Variable",
    "Inputs",
    # Utility nodes
    "Clock",
    "StepCounter",
    "Logger",
    # Enums and status
    "ResolveStatus",
    # Context managers
    "symbolic_mode",
    # Version and metadata
    "get_version",
    "metadata",
]
