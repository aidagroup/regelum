"""Mass-spring-damper system."""

import numpy as np
from regelum.utils import rg
from regelum.node.base import Node
from typing import Callable


class MassSpringDamper(Node):
    """Mass-spring-damper system.

    Models a mass connected to a fixed point by a spring and damper:
    - Linear spring force: -kx
    - Linear damping force: -bẋ
    - External force input: u

    State:
        x[0]: Position (x) [m]
        x[1]: Velocity (ẋ) [m/s]

    Input:
        u: External force [N]

    Parameters:
        mass: Mass (1.0 kg)
        spring_k: Spring constant (10.0 N/m)
        damping_b: Damping coefficient (0.5 N⋅s/m)

    Dynamics:
        mẍ = -kx - bẋ + u
    """

    def __init__(
        self,
        control_signal_name: str,
        initial_state: np.ndarray | None = None,
        state_reset_modifier: Callable | None = None,
        params: dict | None = None,
    ):
        """Initialize mass-spring-damper system.

        Args:
            control_signal_name: Name of control input
            initial_state: Initial [position, velocity]
            state_reset_modifier: Function to modify reset state
            params: Optional parameter overrides
        """
        if initial_state is None:
            initial_state = np.array([1.0, 0.0])  # Start with displacement

        super().__init__(
            inputs=[control_signal_name],
            step_size=0.01,
            is_continuous=True,
            name="mass_spring_damper",
            is_root=True,
        )

        # System parameters
        default_params = {
            "mass": 1.0,  # Mass [kg]
            "spring_k": 10.0,  # Spring constant [N/m]
            "damping_b": 0.5,  # Damping coefficient [N⋅s/m]
        }
        params = params or default_params
        self.mass = params["mass"]
        self.spring_k = params["spring_k"]
        self.damping_b = params["damping_b"]

        # Initialize state
        self.state = self.define_variable(
            "state",
            value=initial_state,
            metadata={"shape": (2,)},
            reset_modifier=state_reset_modifier,
        )

    def state_transition_map(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute state derivatives.

        Args:
            x: State vector [position, velocity]
            u: Control input [force]

        Returns:
            State derivatives [dx/dt, dv/dt]
        """
        position, velocity = x[0], x[1]
        force = u[0] if isinstance(u, np.ndarray) else u

        # Spring force + Damping force + External force
        acceleration = (
            -self.spring_k * position - self.damping_b * velocity + force
        ) / self.mass

        return rg.vstack([velocity, acceleration])

    def step(self) -> None:
        """Execute one simulation step."""
        if self.resolved_inputs is None:
            return

        action = self.resolved_inputs.find(self.inputs.inputs[0]).value
        self.state.value += (
            self.state_transition_map(self.state.value, action) * self.step_size
        ).reshape(-1)
