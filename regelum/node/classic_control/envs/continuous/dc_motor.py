"""DC motor system."""

import numpy as np
from regelum.utils import rg
from regelum.node.base import Node


class DCMotor(Node):
    """DC motor with armature control.

    Models a DC motor with:
    - Electrical dynamics (RL circuit)
    - Mechanical dynamics (inertia + damping)
    - Back-EMF coupling

    State:
        x[0]: Angular position (θ) [rad]
        x[1]: Angular velocity (ω) [rad/s]
        x[2]: Armature current (i) [A]

    Input:
        u: Armature voltage [V]

    Parameters:
        J: Rotor inertia (0.01 kg⋅m²)
        b: Viscous friction (0.1 N⋅m⋅s)
        K: Motor constant (0.01 N⋅m/A)
        R: Armature resistance (1 Ω)
        L: Armature inductance (0.5 H)

    Dynamics:
        Jω̇ = Ki - bω
        Li̇ = -Ri - Kω + u
    """

    def __init__(
        self,
        control_signal_name: str,
        initial_state: np.ndarray | None = None,
        state_reset_modifier=None,
        params: dict | None = None,
    ):
        """Initialize DC motor system.

        Args:
            control_signal_name: Name of control input
            initial_state: Initial [theta, omega, current]
            state_reset_modifier: Function to modify reset state
            params: Optional parameter overrides
        """
        if initial_state is None:
            initial_state = np.array([0.0, 0.0, 0.0])

        super().__init__(
            inputs=[control_signal_name],
            step_size=0.01,
            is_continuous=True,
            name="dc_motor",
            is_root=True,
        )

        # System parameters
        default_params = {
            "J": 0.01,  # Rotor inertia [kg⋅m²]
            "b": 0.1,  # Viscous friction [N⋅m⋅s]
            "K": 0.01,  # Motor constant [N⋅m/A]
            "R": 1.0,  # Armature resistance [Ω]
            "L": 0.5,  # Armature inductance [H]
        }
        params = params or default_params
        self.J = params["J"]
        self.b = params["b"]
        self.K = params["K"]
        self.R = params["R"]
        self.L = params["L"]

        # Initialize state
        self.state = self.define_variable(
            "state",
            value=initial_state,
            metadata={"shape": (3,)},
            reset_modifier=state_reset_modifier,
        )

    def state_transition_map(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute state derivatives.

        Args:
            x: State vector [theta, omega, current]
            u: Control input [voltage]

        Returns:
            State derivatives [dθ/dt, dω/dt, di/dt]
        """
        _, omega, current = x[0], x[1], x[2]
        voltage = u[0] if isinstance(u, np.ndarray) else u

        # Mechanical dynamics
        angular_acc = (self.K * current - self.b * omega) / self.J

        # Electrical dynamics
        current_derivative = (-self.R * current - self.K * omega + voltage) / self.L

        return rg.vstack([omega, angular_acc, current_derivative])

    def step(self) -> None:
        """Execute one simulation step."""
        if self.resolved_inputs is None:
            return

        action = self.resolved_inputs.find(self.inputs.inputs[0]).value
        self.state.value += (
            self.state_transition_map(self.state.value, action) * self.step_size
        ).reshape(-1)
