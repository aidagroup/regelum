"""Kinematic point environment."""

from typing import Any, Callable

from regelum import Node
from regelum.node.core.types import NumericArray
import numpy as np


class KinematicPoint(Node):
    """System representing a simple 2D kinematic point."""

    def __init__(
        self,
        control_signal_name: str,
        state_reset_modifier: Callable[[Any], Any] = None,
    ):
        """Initialize kinematic point environment."""
        super().__init__(
            is_root=True,
            is_continuous=True,
            inputs=[control_signal_name],
            name="kinematic-point",
        )
        self.control_signal_name = control_signal_name
        self.state = self.define_variable(
            "state",
            value=np.ones(2),
            shape=(2,),
            reset_modifier=state_reset_modifier,
        )

    def state_transition_map(self, x: NumericArray, u: NumericArray) -> NumericArray:
        """Compute right-hand side of kinematic point.

        Args:
            x: Current state [x, y].
            u: Current control inputs [v_x, v_y].

        Returns:
            State derivatives [dx/dt, dy/dt].
        """
        return u

    def step(self) -> None:
        action = self.resolved_inputs.find(self.inputs.inputs[0]).value
        self.state.value += (
            self.state_transition_map(self.state.value, action) * self.step_size
        ).reshape(-1)
