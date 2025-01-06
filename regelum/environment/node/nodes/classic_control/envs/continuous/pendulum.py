"""Pendulum is a node that represents a pendulum system."""

from typing import Any, Callable

from regelum import Node
import numpy as np
from regelum.utils import rg


class Pendulum(Node):
    """Pendulum is a node that represents a pendulum system."""

    def __init__(
        self,
        control_signal_name: str,
        state_reset_modifier: Callable[[Any], Any] = None,
    ):
        """Initialize the Pendulum node.

        Args:
            control_signal_name (str): The name of the control signal input.
            state_reset_modifier (Callable[[Any], Any]): A function that modifies the reset state.
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
        self.state = self.define_variable(
            "pendulum_state",
            value=np.array([np.pi, 0]),
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
        action = self.resolved_inputs.find(self.control_signal_name).value
        self.state.value += (
            self.state_transition_map(self.state.value, action) * self.step_size
        ).reshape(-1)
