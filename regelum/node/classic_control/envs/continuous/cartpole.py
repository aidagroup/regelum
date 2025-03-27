"""Cart-pole system (inverted pendulum on a cart)."""

import numpy as np
from regelum.utils import rg
from regelum.node.base import Node
from typing import Callable


class CartPole(Node):
    """Cart-pole system without friction.

    Models an inverted pendulum mounted on a cart that moves horizontally.
    The system is underactuated with force input applied only to the cart.

    State:
        x[0]: pendulum angle (θ) [rad], zero is upward
        x[1]: cart position (x) [m]
        x[2]: pendulum angular velocity (θ̇) [rad/s]
        x[3]: cart velocity (ẋ) [m/s]

    Input:
        u: horizontal force on cart [N]

    Parameters:
        m_c: Cart mass (0.1 kg)
        m_p: Pendulum mass (2.0 kg)
        l: Pendulum length (0.5 m)
        g: Gravity (9.81 m/s²)

    Dynamics:
        θ̈ = (g sin θ - cos θ (F + m_p l θ̇² sin θ)/(m_c + m_p)) / l
             / (4/3 - m_p cos²θ/(m_c + m_p))
        ẍ = (F + m_p l (θ̇² sin θ - θ̈ cos θ))/(m_c + m_p)
    """

    def __init__(
        self,
        control_signal_name: str,
        initial_state: np.ndarray | None = None,
        state_reset_modifier: Callable | None = None,
        params: dict | None = None,
    ):
        """Initialize cart-pole system.

        Args:
            control_signal_name: Name of control input
            initial_state: Initial [theta, x, theta_dot, x_dot]
            state_reset_modifier: Function to modify reset state
            params: Optional parameter overrides
        """
        if initial_state is None:
            initial_state = np.array([np.pi, 0.0, 0.0, 0.0])

        super().__init__(
            inputs=[control_signal_name],
            step_size=0.01,
            is_continuous=True,
            name="cartpole",
            is_root=True,
        )

        # System parameters
        default_params = {
            "m_c": 0.1,  # Cart mass [kg]
            "m_p": 2.0,  # Pendulum mass [kg]
            "l": 0.5,  # Pendulum length [m]
            "g": 9.81,  # Gravity [m/s²]
        }
        params = params or default_params
        self.m_c = params["m_c"]
        self.m_p = params["m_p"]
        self.l = params["l"]
        self.g = params["g"]

        # Initialize state
        self.state = self.define_variable(
            "state",
            value=initial_state,
            metadata={"shape": (4,)},
            reset_modifier=state_reset_modifier,
        )

    def state_transition_map(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute state derivatives.

        Args:
            x: State vector [theta, x, theta_dot, x_dot]
            u: Control input [force]

        Returns:
            State derivatives [dθ/dt, dx/dt, dθ̇/dt, dẋ/dt]
        """
        theta, _, theta_dot, x_dot = x[0], x[1], x[2], x[3]
        force = u[0] if isinstance(u, np.ndarray) else u

        sin_theta = rg.sin(theta)
        cos_theta = rg.cos(theta)

        # Compute angular acceleration
        theta_acc = (
            (
                self.g * sin_theta
                - cos_theta
                * (force + self.m_p * self.l * theta_dot**2 * sin_theta)
                / (self.m_c + self.m_p)
            )
            / self.l
            / (4 / 3 - self.m_p * cos_theta**2 / (self.m_c + self.m_p))
        )

        # Compute cart acceleration
        x_acc = (
            force
            + self.m_p * self.l * (theta_dot**2 * sin_theta - theta_acc * cos_theta)
        ) / (self.m_c + self.m_p)

        return rg.vstack([theta_dot, x_dot, theta_acc, x_acc])

    def step(self) -> None:
        """Execute one simulation step."""
        if self.resolved_inputs is None:
            return

        action = self.resolved_inputs.find(self.inputs.inputs[0]).value
        self.state.value += (
            self.state_transition_map(self.state.value, action) * self.step_size
        ).reshape(-1)
