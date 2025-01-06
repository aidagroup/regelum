"""Logging nodes implementation module."""

from __future__ import annotations
from typing import List

import numpy as np

from regelum.environment.node.nodes.base import Node
from regelum.utils.logger import logger


class Clock(Node):
    """Time management node."""

    def __init__(self, fundamental_step_size: float) -> None:
        """Initialize Clock node."""
        super().__init__(
            step_size=fundamental_step_size,
            is_continuous=False,
            is_root=False,
            name="clock",
        )
        self.fundamental_step_size = fundamental_step_size
        self.time = self.define_variable("time", value=0.0)

    def step(self) -> None:
        """Increment time by fundamental step size."""
        assert isinstance(self.time.value, float), "Time must be a float"
        self.time.value += self.fundamental_step_size

    def _original_reset(self) -> None:
        """Original reset implementation."""
        self.reset(apply_reset_modifier=False)


class StepCounter(Node):
    """Counts steps in the simulation."""

    def __init__(self, nodes: List[Node], start_count: int = 0) -> None:
        """Initialize StepCounter node."""
        step_sizes = [
            node.step_size
            for node in nodes
            if not node.is_continuous and node.step_size is not None
        ]
        if not step_sizes:
            raise ValueError("No non-continuous nodes with step size provided")

        super().__init__(
            step_size=min(step_sizes),
            is_continuous=False,
            is_root=False,
            name="step_counter",
        )
        self.counter = self.define_variable("counter", value=start_count)

    def step(self) -> None:
        """Increment counter."""
        assert isinstance(self.counter.value, int), "Counter must be an integer"
        self.counter.value += 1

    def _original_reset(self) -> None:
        """Original reset implementation."""
        self.reset(apply_reset_modifier=False)


class Logger(Node):
    """Logs variable values during simulation."""

    def __init__(
        self, variables_to_log: List[str], step_size: float, cooldown: float = 0.0
    ) -> None:
        """Initialize Logger node."""
        variables_to_log.extend(["clock_1.time"])
        super().__init__(
            inputs=variables_to_log,
            step_size=step_size,
            is_continuous=False,
            is_root=False,
            name="logger",
        )
        self.cooldown = cooldown
        self.last_log_time = 0.0

    def step(self) -> None:
        """Log current values if cooldown has elapsed."""
        if not self.resolved_inputs:
            return

        time_var = self.resolved_inputs.find("clock_1.time")
        if time_var is None or time_var.value is None:
            return
        current_time = time_var.value
        if current_time - self.last_log_time >= self.cooldown:
            self._log_values()
            self.last_log_time = current_time

    def _log_values(self) -> None:
        """Log current values of tracked variables."""
        if not self.resolved_inputs:
            return

        log_str = "\nCurrent state:"
        for var_name in self.inputs.inputs:
            var = self.resolved_inputs.find(var_name)
            if var is None or var.value is None:
                continue
            value = var.value
            if isinstance(value, np.ndarray):
                value_str = np.array2string(value, precision=4, suppress_small=True)
            elif isinstance(value, float):
                value_str = str(round(value, 3))
            else:
                value_str = str(value)
            log_str += f"\n{var_name}: {value_str}"
        logger.info(log_str)
