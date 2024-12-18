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
import threading
from typing import Dict, Any

__version__ = "1.0.0"
__author__ = "Regelum Team"
__license__ = "MIT"

# Core components
from regelum.environment.node.base_new import (
    Node,
    Graph,
    Variable,
    Inputs,
    ResolveStatus,
)
from regelum.environment.node.library.logging import Clock, StepCounter, Logger
from regelum.environment.node.parallel import ParallelGraph

# Symbolic inference context
_SYMBOLIC_INFERENCE_ACTIVE = threading.local()
_SYMBOLIC_INFERENCE_ACTIVE.value = False


@contextmanager
def symbolic_inference():
    """Enable symbolic inference within a context.

    This context manager temporarily enables symbolic computation mode,
    allowing nodes to work with symbolic variables instead of concrete values.

    Example:
        >>> with symbolic_inference():
        ...     # Code here will use symbolic computation
        ...     graph.step()
    """
    _SYMBOLIC_INFERENCE_ACTIVE.value = True
    try:
        yield
    finally:
        _SYMBOLIC_INFERENCE_ACTIVE.value = False


# Version information
def get_version() -> str:
    """Return the current version of Regelum."""
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
    "symbolic_inference",
    # Version and metadata
    "get_version",
    "metadata",
]
