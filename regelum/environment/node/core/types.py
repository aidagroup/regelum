"""Type definitions for the node system."""

from enum import StrEnum
from typing import Dict, Any, Union, Tuple, TypeAlias, Callable, Literal

import numpy as np
import casadi as cs
import torch

# Type for numeric arrays (numpy, torch, casadi)
NumericArray: TypeAlias = Union[np.ndarray, torch.Tensor, cs.DM, cs.MX]

# Type for variable shapes
Shape: TypeAlias = Union[Tuple[int, ...], None]

# Type for variable values
Value: TypeAlias = Union[NumericArray, float, int, bool, None, Dict[str, Any]]

# Type for metadata keys
MetadataKey: TypeAlias = Literal[
    "initial_value", "symbolic_value", "shape", "reset_modifier", "current_value"
]

# Type for variable metadata
Metadata: TypeAlias = Dict[
    MetadataKey, Union[Value, NumericArray, Shape, Callable[[Any], Any], None]
]


def default_metadata() -> Metadata:
    """Create default metadata structure."""
    return {
        "initial_value": None,
        "symbolic_value": None,
        "shape": None,
        "reset_modifier": None,
        "current_value": None,
    }


# Type for node names
NodeName: TypeAlias = str

# Type for variable names
VarName: TypeAlias = str

# Type for fully qualified names (node_name.var_name)
FullName: TypeAlias = str


class ResolveStatus(StrEnum):
    """Status of the resolve operation."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNDEFINED = "undefined"
