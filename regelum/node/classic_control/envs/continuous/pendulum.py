"""Pendulum environment."""

from typing import Any, Callable

from regelum import Node
from regelum.node.core.types import NumericArray
import numpy as np
from regelum.utils import rg


class Pendulum(Node):
    """Pendulum is a node that represents a pendulum system."""

    def __init__(
        self,
        control_signal_name: str,
        initial_state: NumericArray | None = None,
        state_reset_modifier: Callable[[Any], Any] | None = None,
    ):
        """Initialize the Pendulum node.

        Args:
            control_signal_name: The name of the control signal input.
            initial_state: Initial state of the system [angle, angular_velocity].
            state_reset_modifier: A function that modifies the reset state.
        """
        super().__init__(
            is_root=True,
            is_continuous=True,
            inputs=[control_signal_name],
            name="pendulum",
        )
        self.control_signal_name = control_signal_name
        self.length = 1
        self.mass = 1
        self.gravity_acceleration = 9.81

        if initial_state is None:
            initial_state = np.array([np.pi, 0])

        self.state = self.define_variable(
            "state",
            value=initial_state,
            shape=(2,),
            reset_modifier=state_reset_modifier,
        )

    def state_transition_map(self, x, u):
        pendulum_mpc_control = u

        angle = x[0]
        angular_velocity = x[1]
        torque = pendulum_mpc_control

        d_angle = angular_velocity
        d_angular_velocity = (
            -3 * self.gravity_acceleration / (2 * self.length) * rg.sin(angle)
            + torque / self.mass
        )

        return rg.vstack([d_angle, d_angular_velocity])

    def objective_function(self, x):
        return 4 * x[0] ** 2 + x[1] ** 2

    def step(self):
        action = self.resolved_inputs.find(self.inputs.inputs[0]).value
        self.state.value += (
            self.state_transition_map(self.state.value, action) * self.step_size
        ).reshape(-1)
