from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypedDict,
    TYPE_CHECKING,
)

import casadi as cs
import numpy as np
import torch

from regelum import Node
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
        """Increment counter by 1."""
        assert isinstance(self.counter.value, int), "Counter must be an integer"
        self.counter.value += 1


class Logger(Node):
    """State recording node."""

    def __init__(
        self, variables_to_log: List[str], step_size: float, cooldown: float = 0.0
    ) -> None:
        """Initialize Logger node."""
        super().__init__(
            inputs=["clock_1.time", "step_counter_1.counter"] + variables_to_log,
            step_size=step_size,
            is_continuous=False,
            is_root=False,
            name="logger",
        )
        self.variables_to_log = variables_to_log
        self.cooldown = cooldown
        self.last_log_time = self.define_variable("last_log_time", value=-float("inf"))
        self.log_queue = None

    def step(self) -> None:
        """Log current state if enough time has passed."""
        if self.resolved_inputs is None:
            return
        time_var = self.resolved_inputs.find("clock.time")
        if time_var is None or time_var.value is None:
            return
        current_time = time_var.value
        if current_time - self.last_log_time.value < self.cooldown:
            return

        log_parts = [f"t={current_time:.3f}"]
        for path in self.inputs.inputs:
            var = self.resolved_inputs.find(path)
            if var is None or var.value is None:
                continue
            value = var.value
            formatted_value = (
                f"[{', '.join(f'{v:.3f}' for v in value)}]"
                if isinstance(value, (np.ndarray, list))
                else f"{value:.3f}"
            )
            log_parts.append(f"{path}={formatted_value}")

        log_msg = f"{self.external_name} | " + " | ".join(log_parts)
        if self.log_queue is not None:
            self.log_queue.put(log_msg)
        else:
            logger.info(log_msg)

        self.last_log_time.value = current_time
