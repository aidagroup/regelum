"""Energy-based controllers for underactuated systems."""

from typing import Optional, Tuple
import numpy as np
from regelum.node.base import Node
from regelum import Variable


def hard_switch(signal1: float, signal2: float, condition: bool) -> float:
    """Switch between two signals based on a condition."""
    return signal2 if condition else signal1


class EnergyBasedSwingUpController(Node):
    """Energy-based swing-up controller for pendulum systems.

    Uses energy shaping with friction compensation to swing up and stabilize
    the pendulum. The controller switches between:
    1. Energy shaping with friction compensation for swing-up
    2. PD control for stabilization near the upright position

    The swing-up control law is:
        u = -k_p * sign(ω * E) + k_f * I * ω * |ω|
    where:
        E: System energy
        ω: Angular velocity
        k_p: Energy shaping gain
        k_f: Friction coefficient
        I: Moment of inertia

    Near the upright position, it switches to PD control:
        u = -k_d * ω - k_p * sin(θ)
    """

    def __init__(
        self,
        controlled_state: Variable,
        pendulum_params: dict,
        control_limits: Optional[Tuple[float, float]] = None,
        gain: float = 1.0,
        pd_gains: Optional[Tuple[float, float]] = None,
        switch_threshold: float = 0.95,  # cos(theta) threshold for switching
        step_size: float = 0.01,
        name: str = "energy_swing_up",
    ):
        """Initialize energy-based swing-up controller.

        Args:
            controlled_state: Pendulum state [angle, angular_velocity]
            pendulum_params: Dictionary with mass, length, gravity, friction
            control_limits: Optional control bounds (u_min, u_max)
            gain: Energy shaping gain
            pd_gains: (kp, kd) gains for PD control near upright
            switch_threshold: When to switch to PD control
            step_size: Controller update interval
            name: Controller name
        """
        super().__init__(
            inputs=[controlled_state.full_name],
            step_size=step_size,
            is_continuous=False,
            name=name,
        )

        # System parameters
        self.mass = pendulum_params["mass"]
        self.length = pendulum_params["length"]
        self.gravity = pendulum_params["gravity"]
        self.friction = pendulum_params.get("friction", 0.0)
        self.gain = gain
        self.control_limits = control_limits
        self.pd_gains = pd_gains or (10.0, 1.0)  # Default PD gains
        self.switch_threshold = switch_threshold

        # Derived parameters
        self.moment_inertia = self.mass * self.length**2

        # Initialize control action
        self.action = self.define_variable(
            "action",
            value=np.array([0.0]),
            metadata={"shape": (1,)},
        )

    def _compute_energy(self, state: np.ndarray) -> float:
        """Compute total mechanical energy of the pendulum."""
        if len(state) == 2:
            theta, omega = state
        else:
            theta, omega, x = state

        # Kinetic energy
        T = 0.5 * self.moment_inertia * omega**2

        # Potential energy (zero at upright position)
        V = self.mass * self.gravity * self.length * (np.cos(theta) - 1) / 2

        return T + V

    def step(self) -> None:
        """Execute one control step using energy shaping or PD control."""
        if self.resolved_inputs is None:
            return

        state_var = self.resolved_inputs.find(self.inputs.inputs[0])
        if state_var is None:
            return

        state = state_var.value
        if len(state) == 2:
            theta, omega = state
        else:
            theta, omega, x = state

        # Energy shaping with friction compensation
        energy = self._compute_energy(state)
        energy_control = -self.gain * np.sign(omega * energy)
        friction_comp = self.friction * self.moment_inertia * omega * np.abs(omega)
        swing_up_control = energy_control + friction_comp

        # PD control for stabilization
        pd_control = -self.pd_gains[0] * np.sin(theta) - self.pd_gains[1] * omega

        # Switch between controllers based on angle
        u = hard_switch(
            signal1=swing_up_control,
            signal2=pd_control,
            condition=np.cos(theta) >= self.switch_threshold,
        )

        # Apply control limits
        if self.control_limits is not None:
            u = np.clip(u, self.control_limits[0], self.control_limits[1])

        self.action.value = np.array([u])
