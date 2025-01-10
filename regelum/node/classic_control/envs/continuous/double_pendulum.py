"""Double pendulum system implementation.

The double pendulum consists of two pendulums attached end to end,
with the second pendulum's pivot point being the end of the first pendulum.

State vector: [θ₁, θ₂, ω₁, ω₂]
where:
- θ₁: angle of first pendulum from vertical
- θ₂: angle of second pendulum from vertical
- ω₁: angular velocity of first pendulum
- ω₂: angular velocity of second pendulum

Control input: [τ₁, τ₂]
- τ₁: torque applied to first pendulum
- τ₂: torque applied to second pendulum
"""

import numpy as np
from typing import Optional, Tuple
from regelum.node.base import Node
from regelum.node.core.types import NumericArray
from regelum.utils import rg


class DoublePendulum(Node):
    """Double pendulum system with control inputs.

    The system implements the full nonlinear dynamics of a double pendulum
    with optional control torques applied at both joints.

    Parameters:
        m1: Mass of first pendulum
        m2: Mass of second pendulum
        l1: Length of first pendulum
        l2: Length of second pendulum
        g: Gravitational acceleration
        control_signal_name: Name of control input variable
        initial_state: Initial state vector [θ₁, θ₂, ω₁, ω₂]
    """

    def __init__(
        self,
        m1: float = 1.0,
        m2: float = 1.0,
        l1: float = 1.0,
        l2: float = 1.0,
        g: float = 9.81,
        control_signal_name: Optional[str] = None,
        initial_state: Optional[NumericArray] = None,
        step_size: float = 0.01,
    ):
        """Initialize double pendulum system."""
        inputs = [control_signal_name] if control_signal_name else []
        super().__init__(
            inputs=inputs,
            step_size=step_size,
            is_continuous=True,
            name="double_pendulum",
        )

        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g

        if initial_state is None:
            initial_state = np.array([np.pi / 4, np.pi / 4, 0.0, 0.0])

        self.state = self.define_variable(
            name="state",
            value=initial_state,
            shape=(4,),
        )

    def state_transition_map(
        self, state: NumericArray, control: Optional[NumericArray] = None
    ) -> NumericArray:
        """Compute state derivatives for the double pendulum system.

        Implements the full nonlinear equations of motion for the double pendulum.

        Args:
            state: Current state [θ₁, θ₂, ω₁, ω₂]
            control: Optional control torques [τ₁, τ₂]

        Returns:
            State derivatives [dθ₁/dt, dθ₂/dt, dω₁/dt, dω₂/dt]
        """
        theta1, theta2, omega1, omega2 = state[0], state[1], state[2], state[3]

        # Default zero control if none provided
        if control is None:
            tau1, tau2 = 0.0, 0.0
        else:
            tau1, tau2 = control[0], control[1]

        # Compute common terms
        c = rg.cos(theta1 - theta2)
        s = rg.sin(theta1 - theta2)

        # Mass matrix terms
        m11 = (
            (self.m1 + self.m2) * self.l1**2
            + self.m2 * self.l2**2
            + 2 * self.m2 * self.l1 * self.l2 * c
        )
        m12 = self.m2 * self.l2**2 + self.m2 * self.l1 * self.l2 * c
        m21 = m12
        m22 = self.m2 * self.l2**2

        # Force terms
        f1 = (
            -self.m2 * self.l1 * self.l2 * s * omega2**2
            - 2 * self.m2 * self.l1 * self.l2 * s * omega1 * omega2
            - (self.m1 + self.m2) * self.g * self.l1 * rg.sin(theta1)
            + tau1
        )

        f2 = (
            self.m2 * self.l1 * self.l2 * s * omega1**2
            - self.m2 * self.g * self.l2 * rg.sin(theta2)
            + tau2
        )

        # Compute determinant for matrix inversion
        det = m11 * m22 - m12 * m21

        # Compute angular accelerations
        alpha1 = (m22 * f1 - m12 * f2) / det
        alpha2 = (-m21 * f1 + m11 * f2) / det

        return rg.vstack([omega1, omega2, alpha1, alpha2])

    def get_cartesian_coords(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get Cartesian coordinates of both pendulum masses.

        Returns:
            Tuple of (x1,y1), (x2,y2) coordinates for visualization
        """
        theta1, theta2 = self.state.value[0:2]

        x1 = self.l1 * np.sin(theta1)
        y1 = -self.l1 * np.cos(theta1)

        x2 = x1 + self.l2 * np.sin(theta2)
        y2 = y1 - self.l2 * np.cos(theta2)

        return (x1, y1), (x2, y2)

    def step(self, control: Optional[NumericArray] = None) -> None:
        """Step the double pendulum system."""
        self.state.value += (
            self.state_transition_map(self.state.value, control).reshape(-1)
            * self.step_size
        )
