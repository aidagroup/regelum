"""Type definitions for the node system."""

from enum import StrEnum
from typing import Dict, Any, Union, Tuple, TypeAlias, Callable, Literal

import numpy as np
import casadi as cs
import torch

NumericArray: TypeAlias = Union[np.ndarray, torch.Tensor, cs.DM, cs.MX]

Shape: TypeAlias = Union[Tuple[int, ...], None]

Value: TypeAlias = Union[NumericArray, float, int, bool, None, Dict[str, Any]]

MetadataKey: TypeAlias = Literal[
    "initial_value", "symbolic_value", "shape", "reset_modifier", "current_value"
]

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


NodeName: TypeAlias = str

VarName: TypeAlias = str

FullName: TypeAlias = str


class ResolveStatus(StrEnum):
    """Status of the resolve operation."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNDEFINED = "undefined"
