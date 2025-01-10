"""Pendulum system with nonlinear friction."""

import numpy as np
from regelum.utils import rg
from regelum.node.classic_control.envs.continuous import Pendulum


class PendulumWithFriction(Pendulum):
    """Pendulum with quadratic friction.

    Models a physical pendulum with parameters similar to Quanser Rotary Pendulum.
    The system includes quadratic damping (friction proportional to ω|ω|).

    State:
        x[0]: angle (θ) [rad], zero is upward
        x[1]: angular velocity (ω) [rad/s]

    Parameters:
        mass: 0.127 kg
        length: 0.337 m
        gravity: 9.81 m/s²
        friction: 0.08 N⋅m⋅s²/rad²

    Dynamics:
        θ̈ = (mgL/2)sin(θ) + u - kω|ω|
        where k is the friction coefficient
    """

    def __init__(
        self,
        control_signal_name: str,
        initial_state: np.ndarray | None = None,
        state_reset_modifier=None,
        friction_coeff: float = 0.08,
    ):
        """Initialize pendulum with friction.

        Args:
            control_signal_name: Name of control input
            initial_state: Initial [angle, angular_velocity]
            state_reset_modifier: Function to modify reset state
            friction_coeff: Quadratic friction coefficient
        """
        super().__init__(
            control_signal_name=control_signal_name,
            initial_state=initial_state,
            state_reset_modifier=state_reset_modifier,
        )

        # Override default parameters with Quanser-like values
        self.mass = 0.127  # kg
        self.length = 0.337  # m
        self.gravity_acceleration = 9.81  # m/s²
        self.friction_coeff = friction_coeff  # N⋅m⋅s²/rad²

    def state_transition_map(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute state derivatives including friction.

        Args:
            x: State vector [angle, angular_velocity]
            u: Control input [torque]

        Returns:
            State derivatives [dθ/dt, dω/dt]
        """
        theta, omega = x[0], x[1]
        torque = u[0] if isinstance(u, np.ndarray) else u

        # Angular acceleration with friction
        alpha = (
            self.gravity_acceleration * self.mass * self.length * rg.sin(theta) / 2
            + torque
        ) / (self.mass * self.length**2) - self.friction_coeff * omega**2 * rg.sign(
            omega
        )

        return rg.vstack([omega, alpha])
