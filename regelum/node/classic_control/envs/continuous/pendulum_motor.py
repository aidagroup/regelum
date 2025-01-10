"""Pendulum system with motor dynamics."""

import numpy as np
from regelum.utils import rg
from regelum.node.classic_control.envs.continuous import Pendulum


class PendulumWithMotor(Pendulum):
    """Pendulum with DC motor dynamics.

    Models a physical pendulum driven by a DC motor with:
    - First-order motor dynamics
    - Additional motor inertia
    - Parameters similar to Quanser setup

    State:
        x[0]: angle (θ) [rad], zero is upward
        x[1]: angular velocity (ω) [rad/s]
        x[2]: motor torque (τ) [N⋅m]

    Input:
        u: motor command [N⋅m/s]

    Parameters:
        mass: 0.127 kg (pendulum)
        length: 0.337 m
        gravity: 9.81 m/s²
        motor_mass: 0.1 kg
        motor_radius: 0.04 m
        motor_time_const: 0.05 s

    Dynamics:
        θ̈ = (mgL/2)sin(θ) + τ)/(J_p + J_m)
        τ̇ = (u - τ)/T
    where:
        J_p: pendulum inertia
        J_m: motor inertia
        T: motor time constant
    """

    def __init__(
        self,
        control_signal_name: str,
        initial_state: np.ndarray | None = None,
        state_reset_modifier=None,
        motor_params: dict | None = None,
    ):
        """Initialize pendulum with motor.

        Args:
            control_signal_name: Name of control input
            initial_state: Initial [angle, angular_velocity, torque]
            state_reset_modifier: Function to modify reset state
            motor_params: Optional motor parameters override
        """
        if initial_state is None:
            initial_state = np.array([np.pi, 0.0, 0.0])

        super().__init__(
            control_signal_name=control_signal_name,
            initial_state=initial_state,
            state_reset_modifier=state_reset_modifier,
        )

        # Override with Quanser-like parameters
        self.mass = 0.127  # kg
        self.length = 0.337  # m
        self.gravity_acceleration = 9.81  # m/s²

        # Motor parameters
        default_motor = {
            "mass": 0.1,  # kg
            "radius": 0.04,  # m
            "time_const": 0.05,  # s
        }
        motor_params = motor_params or default_motor
        self.motor_mass = motor_params["mass"]
        self.motor_radius = motor_params["radius"]
        self.motor_time_const = motor_params["time_const"]

    def motor_moment_inertia(self) -> float:
        """Compute motor moment of inertia."""
        return self.motor_mass * self.motor_radius**2 / 2

    def state_transition_map(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Compute state derivatives including motor dynamics.

        Args:
            x: State vector [angle, angular_velocity, torque]
            u: Control input [motor_command]

        Returns:
            State derivatives [dθ/dt, dω/dt, dτ/dt]
        """
        theta, omega, torque = x
        motor_command = u[0] if isinstance(u, np.ndarray) else u

        # Total moment of inertia
        J_total = self.mass * self.length**2 + self.motor_moment_inertia()

        # Angular acceleration
        alpha = (
            self.gravity_acceleration * self.mass * self.length * rg.sin(theta) / 2
            + torque
        ) / J_total

        # Motor dynamics
        torque_derivative = (motor_command - torque) / self.motor_time_const

        return rg.vstack([omega, alpha, torque_derivative])
