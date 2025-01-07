"""Variable implementation module."""

from __future__ import annotations
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Optional, Any, Dict

import casadi as cs
import numpy as np
import torch

from regelum.node.interfaces.base import IVariable
from regelum.node.core.globals import _SYMBOLIC_INFERENCE_ACTIVE
from .types import (
    Value,
    Metadata,
    Shape,
    NodeName,
    VarName,
    NumericArray,
    default_metadata,
)


@dataclass(slots=True)
class Variable(IVariable):
    """Implementation of IVariable for the node system."""

    name: VarName
    metadata: Metadata = field(default_factory=default_metadata)
    _node_name: NodeName = field(default="")

    def __post_init__(self) -> None:
        """Initialize metadata with initial value if not present."""
        self.metadata.setdefault("current_value", None)
        if (
            "initial_value" not in self.metadata
            or self.metadata["initial_value"] is None
        ):
            self.metadata["initial_value"] = deepcopy(self.metadata["current_value"])

    @property
    def value(self) -> Optional[Value]:
        """Get the current value."""
        val = self.metadata["current_value"]
        return (
            val
            if isinstance(val, (np.ndarray, torch.Tensor, cs.DM, float, int, bool))
            or val is None
            else None
        )

    @value.setter
    def value(self, val: Optional[Value]) -> None:
        """Set the current value."""
        self.metadata["current_value"] = val

    def reset(self, *, apply_reset_modifier: bool = True) -> None:
        """Reset variable to its initial state."""
        if (
            apply_reset_modifier
            and "reset_modifier" in self.metadata
            and self.metadata["reset_modifier"] is not None
            and callable(self.metadata["reset_modifier"])
        ):
            self.metadata["current_value"] = deepcopy(
                self.metadata["reset_modifier"](self.metadata["initial_value"])
            )
        else:
            self.metadata["current_value"] = deepcopy(self.metadata["initial_value"])

    def get_value(self) -> Any:
        """Get current value, either symbolic or actual based on context."""
        return (
            self.to_casadi_symbolic()
            if getattr(_SYMBOLIC_INFERENCE_ACTIVE, "value", False)
            else self.value
        )

    @property
    def full_name(self) -> str:
        """Get fully qualified name."""
        return f"{self.node_name}.{self.name}"

    def set_new_value(self, value: Any) -> None:
        """Set new value and update initial value."""
        self.value = value
        self.metadata["initial_value"] = deepcopy(value)

    def to_casadi_symbolic(self) -> Optional[NumericArray]:
        """Create or return symbolic representation."""
        sym_val = self.metadata["symbolic_value"]
        if sym_val is None:
            shape = self._infer_shape()
            if shape:
                self.metadata["symbolic_value"] = cs.MX.sym(str(self.name), *shape)  # type: ignore
                sym_val = self.metadata["symbolic_value"]
        return (
            sym_val if isinstance(sym_val, (np.ndarray, torch.Tensor, cs.DM)) else None
        )

    def _infer_shape(self) -> Optional[Shape]:
        """Infer shape from value or metadata."""
        shape_val = self.metadata["shape"]
        if shape_val and isinstance(shape_val, tuple):
            return shape_val

        if hasattr(self.value, "shape"):
            if isinstance(self.value, (np.ndarray, torch.Tensor, cs.DM)):
                return tuple(int(x) for x in self.value.shape)

        if isinstance(self.value, (int, float, bool)):
            return (1,)

        return None

    @property
    def node_name(self) -> str:
        """Get node name."""
        return self._node_name

    @node_name.setter
    def node_name(self, new_name: str) -> None:
        """Set node name."""
        self._node_name = new_name

    def __deepcopy__(self, memo: Dict[int, Any]) -> Variable:
        """Create a deep copy of the variable."""
        return Variable(
            name=self.name,
            metadata=deepcopy(self.metadata, memo),
            _node_name=self.node_name,
        )
