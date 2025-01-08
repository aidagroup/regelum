"""Zero-order hold modifier implementation module."""

from typing import Optional, final, Any
import numpy as np
from regelum.node.modifiers.base import StepModifier
from regelum.node.interfaces.node import INode
from regelum.node.core.variable import Variable


@final
class ZeroOrderHold(StepModifier[INode]):
    """Modifier that implements zero-order hold behavior."""

    def __init__(self, node: INode, clock_ref: INode) -> None:
        """Initialize zero-order hold modifier.

        Args:
            node: Node to modify.
            clock_ref: reference to the clock node for timing.
        """
        self._node = node
        self._step_function = node.step
        self._clock = clock_ref
        self.last_update_step: Optional[int] = None

        if node.step_size is not None and clock_ref.step_size is not None:
            self.step_multiplier = round(node.step_size / clock_ref.step_size)
        else:
            raise ValueError("Node and clock must have defined step sizes")

        self.bind_to_node(node)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        time_var = next(
            (var for var in self._clock.variables if var.name == "time"),
            None,
        )
        if not isinstance(time_var, Variable):
            raise ValueError("Clock must have a time variable")

        if self._clock.step_size is None:
            raise ValueError("Clock must have a defined step size")

        time_value = time_var.value
        if time_value is None:
            raise ValueError("Time value must have a value")

        if isinstance(time_value, (np.ndarray, float, int)):
            current_step = round(float(time_value) / self._clock.step_size)
        else:
            raise ValueError("Time value must be numeric")

        if (
            self.last_update_step is None
            or current_step >= self.last_update_step + self.step_multiplier
        ):
            self.step_function(*args, **kwargs)
            self.last_update_step = current_step

    def reset(self, *, apply_reset_modifier: bool = True) -> None:
        self.last_update_step = None
