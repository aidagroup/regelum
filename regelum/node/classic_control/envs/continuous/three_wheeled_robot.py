"""Three wheeled robot environment."""

from typing import Any, Callable

from regelum import Node
from regelum.node.core.types import NumericArray
import numpy as np
from regelum.utils import rg


class ThreeWheeledRobotKinematic(Node):
    """Implements the kinematic three-wheeled robot (a.k.a. Brockett integrator)."""

    def __init__(
        self,
        control_signal_name: str,
        initial_state: NumericArray | None = None,
        state_reset_modifier: Callable[[Any], Any] | None = None,
    ):
        """Initialize three wheeled robot environment.

        Args:
            control_signal_name: Name of the control signal input.
            initial_state: Initial state of the system [x, y, angle].
            state_reset_modifier: Optional function to modify state on reset.
        """
        super().__init__(
            is_root=True,
            is_continuous=True,
            inputs=[control_signal_name],
            name="three-wheeled-robot",
        )
        self.control_signal_name = control_signal_name

        if initial_state is None:
            initial_state = np.zeros(3)

        self.state = self.define_variable(
            "state",
            value=initial_state,
            shape=(3,),
            reset_modifier=state_reset_modifier,
        )

    def state_transition_map(self, x: NumericArray, u: NumericArray) -> NumericArray:
        """Compute right-hand side of the dynamic system.

        Args:
            x: Current state [x, y, angle].
            u: Current control inputs [velocity, angular_velocity].

        Returns:
            State derivatives [dx/dt, dy/dt, dangle/dt].
        """
        Dstate = rg.zeros(3, prototype=(x, u))

        Dstate[0] = u[0] * rg.cos(x[2])
        Dstate[1] = u[0] * rg.sin(x[2])
        Dstate[2] = u[1]

        return Dstate

    def step(self) -> None:
        action = self.resolved_inputs.find(self.inputs.inputs[0]).value
        self.state.value += (
            self.state_transition_map(self.state.value, action) * self.step_size
        ).reshape(-1)
