"""PID controller for a continuous system."""

from typing import Optional
from regelum.node.nodes.base import Node
from regelum.node.core.variable import Variable

import numpy as np


class PIDControllerBase(Node):
    """PID controller for a continuous system."""

    def __init__(
        self,
        controlled_state: Variable,
        idx_controlled_state: int,
        kp: float,
        ki: float,
        kd: float,
        setpoint: Optional[Variable] = None,
        step_size: float = 0.1,
    ):
        """Initialize PID controller."""
        self.controlled_state = controlled_state
        self.idx_controlled_state = idx_controlled_state
        self.setpoint = setpoint
        inputs = (
            [controlled_state.full_name]
            if setpoint is None
            else [controlled_state.full_name, setpoint.full_name]
        )
        super().__init__(
            name="pid_controller",
            inputs=inputs,
            step_size=step_size,
            is_continuous=False,
        )
        if setpoint is None:
            self.setpoint = self.define_variable(
                "setpoint",
                value=np.zeros((1,)),
                metadata={"shape": controlled_state.metadata["shape"]},
            )
        else:
            self.setpoint = setpoint
            inputs.append(setpoint.full_name)

        self.previous_state = self.define_variable(
            "previous_state",
            value=controlled_state.value,
        )
        self.previous_error = self.define_variable(
            "previous_error",
            value=np.zeros((1,)),
        )
        self.control_signal = self.define_variable(
            "control_signal",
            value=np.zeros((1,)),
        )
        self.error_integral = self.define_variable(
            "error_integral",
            value=np.zeros((1,)),
        )
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def step(self) -> None:
        current_error = (
            self.controlled_state.value[self.idx_controlled_state] - self.setpoint.value
        )
        error_derivative = (current_error - self.previous_error.value) / self.step_size
        self.previous_error.value = current_error

        if self.ki != 0:
            self.error_integral.value += current_error * self.step_size

        self.control_signal.value = -(
            self.kp * current_error
            + self.ki * self.error_integral.value
            + self.kd * error_derivative
        )
        self.previous_state.value = self.controlled_state.value
