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
        initial_state: NumericArray | None = None,
        state_reset_modifier: Callable[[Any], Any] | None = None,
    ):
        """Initialize kinematic point environment.

        Args:
            control_signal_name: Name of the control signal input.
            initial_state: Initial state of the system [x, y].
            state_reset_modifier: Optional function to modify state on reset.
        """
        super().__init__(
            is_root=True,
            is_continuous=True,
            inputs=[control_signal_name],
            name="kinematic-point",
        )
        self.control_signal_name = control_signal_name

        if initial_state is None:
            initial_state = np.ones(2)

        self.state = self.define_variable(
            "state",
            value=initial_state,
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
