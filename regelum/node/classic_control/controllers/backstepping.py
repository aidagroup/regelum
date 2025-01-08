"""Backstepping controller for pendulum with motor dynamics.

This module implements a hybrid backstepping controller that combines:
1. Energy-based control for swing-up
2. PD control for stabilization
3. Backstepping for motor dynamics compensation

The controller switches between swing-up and stabilization based on
the pendulum's state relative to the upright position.
"""

import numpy as np
from typing import List
from regelum.node.base import Node
from regelum import Variable
from regelum.node.core.types import NumericArray
from regelum.node.classic_control.envs.continuous.pendulum_motor import (
    PendulumWithMotor,
)


def hard_switch(signal1: float, signal2: float, condition: bool) -> float:
    """Switch between two signals based on a condition.

    Args:
        signal1: First control signal (used when condition is True)
        signal2: Second control signal (used when condition is False)
        condition: Switching condition

    Returns:
        Selected control signal
    """
    return signal1 if condition else signal2


class PendulumBackstepping(Node):
    """Backstepping controller for pendulum with motor.

    Implements a hybrid control strategy:
    1. Energy-based swing-up with backstepping for large deviations
    2. PD control near the upright position

    The controller uses backstepping to handle the motor dynamics during
    swing-up, ensuring proper tracking of the energy-based control signal.

    Attributes:
        energy_gain: Gain for energy-based swing-up control
        backstepping_gain: Gain for motor dynamics compensation
        switch_loc: Switching threshold for swing-up/stabilization
        pd_coeffs: PD control coefficients [kp, kd]
        action_min: Minimum control input
        action_max: Maximum control input
        system: Reference to the pendulum-motor system
    """

    def __init__(
        self,
        state_variable: Variable,
        energy_gain: float,
        backstepping_gain: float,
        switch_loc: float,
        pd_coeffs: List[float],
        action_min: float,
        action_max: float,
        system: PendulumWithMotor,
        step_size: float = 0.01,
    ):
        """Initialize backstepping controller.

        Args:
            state_variable: System state to control
            energy_gain: Gain for energy-based control
            backstepping_gain: Gain for backstepping compensation
            switch_loc: Threshold for control switching
            pd_coeffs: PD control coefficients [kp, kd]
            action_min: Minimum control input
            action_max: Maximum control input
            system: Reference to pendulum-motor system
            step_size: Controller update interval
        """
        super().__init__(
            inputs=[state_variable.full_name],
            step_size=step_size,
            name="backstepping",
        )

        self.action_min = action_min
        self.action_max = action_max
        self.switch_loc = switch_loc
        self.energy_gain = energy_gain
        self.backstepping_gain = backstepping_gain
        self.pd_coeffs = pd_coeffs
        self.system = system

        # Create action variable
        self.action = self.define_variable(
            "action",
            value=np.zeros(1),
            shape=(1,),
        )

    def compute_control(self, state: NumericArray) -> float:
        """Compute control action using backstepping approach.

        Args:
            state: System state [angle, angular_velocity, torque]

        Returns:
            Control action (motor command)
        """
        # Extract state components
        angle = state[0]
        angle_vel = state[1]
        torque = state[2]

        # Get system parameters
        mass = self.system.mass
        grav_const = self.system.gravity_acceleration
        length = self.system.length

        # Total moment of inertia (pendulum + motor)
        J_total = mass * length**2 + self.system.motor_moment_inertia()

        # Compute total mechanical energy
        energy_total = (
            mass * grav_const * length * (np.cos(angle) - 1) / 2
            + 0.5 * J_total * angle_vel**2
        )

        # Energy-based swing-up control
        energy_control_action = -self.energy_gain * np.sign(angle_vel * energy_total)

        # Backstepping compensation for motor dynamics
        backstepping_action = torque - self.backstepping_gain * (
            torque - energy_control_action
        )

        # PD control for stabilization
        action_pd = -self.pd_coeffs[0] * np.sin(angle) - self.pd_coeffs[1] * angle_vel

        # Switch between controllers based on state
        action = hard_switch(
            signal1=backstepping_action,
            signal2=action_pd,
            condition=(np.cos(angle) - 1) ** 2 + angle_vel**2 >= self.switch_loc,
        )

        # Clip control action
        return np.clip(action, self.action_min, self.action_max)

    def step(self) -> None:
        """Execute one control step."""
        if self.resolved_inputs is None:
            return

        # Get current state
        state = self.resolved_inputs.find(self.inputs.inputs[0]).value

        # Compute and set control action
        self.action.value = np.array([self.compute_control(state)])
