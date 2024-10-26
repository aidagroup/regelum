"""Type Specifications for the Regelum Framework.

This module contains type aliases for array types that are used throughout the Regelum framework.
These type specifications are utilized to ensure that the framework can handle different kinds
of array-like data structures consistently across its various components, providing flexibility
and compatibility with major numerical computing libraries.

The type specifications are designed to support the integration of Numpy arrays, PyTorch tensors,
and CasADi matrix types, making it easier to develop reinforcement learning and optimal control
algorithms that can leverage the strengths of each of these libraries.
"""

import numpy as np
import casadi as cs
from typing import (
    Union,
    Type,
    OrderedDict,
    Dict,
    Protocol
)
import torch as th


RgArrayType = Union[
    Type[np.ndarray],
    Type[th.Tensor],
    Type[cs.DM],
    Type[cs.MX],
]
"""
Type alias for the classes of array types supported by the Regelum framework.

This type alias includes the classes, not instances, of the array types that can be used within
the Regelum framework for RL and optimal control computations. 

Attributes:
    np.array: Represents the class of Numpy array.
    torch.Tensor: Represents the class of PyTorch tensor.
    cs.DM: Represents the class of CasADi's DM matrix type.
    cs.MX: Represents the class of CasADi's MX matrix type.
"""

RgArray = Union[
    np.ndarray,
    th.Tensor,
    cs.DM,
    cs.MX,
]
"""
Type alias for the instances of array types supported by the Regelum framework.

This type alias includes the instances of the array types that can be used within
the Regelum framework for RL and optimal control computations.

Attributes:
    np.array: An instance of a Numpy array.
    torch.Tensor: An instance of a PyTorch tensor.
    cs.DM: An instance of CasADi's DM matrix.
    cs.MX: An instance of CasADi's MX matrix.
"""

Weights = Union[RgArray, OrderedDict[str, th.Tensor]]
"""Type alias for the instances of model's weights. For ModelNN weights are `OrderedDict[str, torch.Tensor]`. For Model weights are `RgArray`."""

LeafStateType = Union[RgArray, Weights]
StateType = Dict[str, Union[LeafStateType, "StateType"]]

class TransitionMapType(Protocol):
    def __call__(self, *states: StateType) -> StateType:
        ...
