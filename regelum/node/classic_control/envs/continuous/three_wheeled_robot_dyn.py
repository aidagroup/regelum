"""Three wheeled robot dynamic environment."""

from typing import Any, Callable

from regelum import Node
from regelum.node.core.types import NumericArray
import numpy as np
from regelum.utils import rg


class ThreeWheeledRobotDynamic(Node):
    """Implements dynamic three-wheeled robot."""

    def __init__(
        self,
        control_signal_name: str,
        initial_state: NumericArray | None = None,
        state_reset_modifier: Callable[[Any], Any] | None = None,
        mass: float = 10.0,
        inertia: float = 1.0,
    ):
        """Initialize dynamic three wheeled robot environment.

        Args:
            control_signal_name: Name of the control signal input.
            initial_state: Initial state [x, y, angle, l_velocity, angular_velocity].
            state_reset_modifier: Optional function to modify state on reset.
            mass: Mass of the robot.
            inertia: Moment of inertia.
        """
        super().__init__(
            is_root=True,
            is_continuous=True,
            inputs=[control_signal_name],
            name="three-wheeled-robot-dynamic",
        )
        self.control_signal_name = control_signal_name
        self.mass = mass
        self.inertia = inertia

        if initial_state is None:
            initial_state = np.zeros(5)

        self.state = self.define_variable(
            "state",
            value=initial_state,
            shape=(5,),
            reset_modifier=state_reset_modifier,
        )

    def state_transition_map(self, x: NumericArray, u: NumericArray) -> NumericArray:
        """Compute right-hand side of the dynamic system.

        Args:
            x: Current state [x, y, angle, l_velocity, angular_velocity].
            u: Current action [Force, Momentum].

        Returns:
            State derivatives [dx/dt, dy/dt, dangle/dt, dl_velocity/dt, dangular_velocity/dt].
        """
        Dstate = rg.zeros(5, prototype=(x, u))

        Dstate[0] = x[3] * rg.cos(x[2])
        Dstate[1] = x[3] * rg.sin(x[2])
        Dstate[2] = x[4]
        Dstate[3] = 1 / self.mass * u[0]
        Dstate[4] = 1 / self.inertia * u[1]

        return Dstate

    def step(self) -> None:
        action = self.resolved_inputs.find(self.inputs.inputs[0]).value
        self.state.value += (
            self.state_transition_map(self.state.value, action) * self.step_size
        ).reshape(-1)
