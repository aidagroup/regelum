"""Discretized pendulum environment."""

from typing import Optional, Tuple
import numpy as np
from regelum.node.base import Node


class DiscretePendulum(Node):
    """Discretized pendulum environment.

    A pendulum system with discretized state and action spaces.
    The state space is discretized into angle and angular velocity bins.
    The action space is discretized torque values.

    State:
        x[0]: Discretized angle index [-π, π] -> [0, n_angles-1]
        x[1]: Discretized velocity index [-8, 8] -> [0, n_velocities-1]

    Actions:
        Discrete indices mapping to torque values in [-2, 2]

    Parameters:
        n_angles: Number of angle discretization bins
        n_velocities: Number of velocity discretization bins
        n_actions: Number of torque discretization bins
    """

    def __init__(
        self,
        control_signal_name: str,
        n_angles: int = 16,
        n_velocities: int = 16,
        n_actions: int = 3,
        initial_state: Optional[np.ndarray] = None,
        step_size: float = 0.05,
    ):
        """Initialize discrete pendulum.

        Args:
            control_signal_name: Name of control input
            n_angles: Number of angle bins
            n_velocities: Number of velocity bins
            n_actions: Number of action bins
            initial_state: Initial [angle_idx, velocity_idx]
            step_size: Time step
        """
        super().__init__(
            inputs=[control_signal_name],
            step_size=step_size,
            is_continuous=False,
            name="discrete_pendulum",
            is_root=True,
        )

        self.n_angles = n_angles
        self.n_velocities = n_velocities
        self.n_actions = n_actions

        # Physical parameters
        self.mass = 1.0
        self.length = 1.0
        self.gravity = 9.81
        self.max_velocity = 8.0
        self.max_torque = 2.0

        if initial_state is None:
            # Start at bottom position with zero velocity
            initial_state = np.array([n_angles // 2, n_velocities // 2])

        # Initialize state
        self.state = self.define_variable(
            "state",
            value=initial_state,
            metadata={"shape": (2,)},
        )

        # Create discretization bins
        self.angle_bins = np.linspace(-np.pi, np.pi, n_angles + 1)
        self.velocity_bins = np.linspace(
            -self.max_velocity, self.max_velocity, n_velocities + 1
        )
        self.action_bins = np.linspace(-self.max_torque, self.max_torque, n_actions)

    def continuous_to_discrete(self, angle: float, velocity: float) -> Tuple[int, int]:
        """Convert continuous state to discrete indices."""
        angle_idx = np.digitize(angle, self.angle_bins) - 1
        velocity_idx = np.digitize(velocity, self.velocity_bins) - 1

        # Clip to valid range
        angle_idx = np.clip(angle_idx, 0, self.n_angles - 1)
        velocity_idx = np.clip(velocity_idx, 0, self.n_velocities - 1)

        return angle_idx, velocity_idx

    def discrete_to_continuous(
        self, angle_idx: int, velocity_idx: int
    ) -> Tuple[float, float]:
        """Convert discrete indices to continuous state."""
        angle = (self.angle_bins[angle_idx] + self.angle_bins[angle_idx + 1]) / 2
        velocity = (
            self.velocity_bins[velocity_idx] + self.velocity_bins[velocity_idx + 1]
        ) / 2
        return angle, velocity

    def step(self) -> None:
        """Execute one step in the environment."""
        if self.resolved_inputs is None:
            return

        # Get discrete action and convert to torque
        action_idx = int(self.resolved_inputs.find(self.inputs.inputs[0]).value)
        torque = self.action_bins[action_idx]

        # Get current continuous state
        angle, velocity = self.discrete_to_continuous(
            int(self.state.value[0]), int(self.state.value[1])
        )

        # Simulate continuous dynamics
        angular_acc = -3 * self.gravity / (2 * self.length) * np.sin(
            angle
        ) + 3 * torque / (self.mass * self.length**2)

        # Euler integration
        new_velocity = velocity + angular_acc * self.step_size
        new_angle = angle + velocity * self.step_size

        # Normalize angle to [-π, π]
        new_angle = ((new_angle + np.pi) % (2 * np.pi)) - np.pi

        # Convert back to discrete state
        angle_idx, velocity_idx = self.continuous_to_discrete(new_angle, new_velocity)
        self.state.value = np.array([angle_idx, velocity_idx])
