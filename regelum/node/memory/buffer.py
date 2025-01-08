"""Buffer for storing data."""

from typing import Dict, List
import numpy as np

from regelum.node.base import Node
from regelum import Variable


class DataBuffer(Node):
    """Buffer for storing data."""

    def __init__(
        self,
        variable_names: List[str],
        buffer_size: int = 1000,
        step_size: float = 0.01,
    ) -> None:
        """Initialize data buffer."""
        assert variable_names is not None
        super().__init__(
            inputs=variable_names + ["step_counter_1.counter"],
            step_size=step_size,
            is_continuous=False,
            name="buffer",
        )
        self.buffer_size = buffer_size
        self._buffers: Dict[str, Variable] = {}

        for full_name in variable_names:
            buffer_var = self.define_variable(
                f"{full_name}@buffer",
                value=None,
            )
            self._buffers[full_name] = buffer_var

    def step(self) -> None:
        if self.resolved_inputs is None:
            return

        for full_name, buffer_var in self._buffers.items():
            input_var = self.resolved_inputs.find(full_name)
            if input_var is None or input_var.value is None:
                continue

            # Initialize buffer if not done yet
            if buffer_var.value is None:
                shape = (self.buffer_size,) + np.array(input_var.value).shape
                buffer_var.value = np.zeros(shape)
                buffer_var.metadata["shape"] = shape

            # Get current buffer index using modulo for circular buffer
            current_idx = (
                int(self.resolved_inputs.find("step_counter_1.counter").value - 1)
                % self.buffer_size
                if self.resolved_inputs.find("step_counter_1.counter") is not None
                else 0
            )

            buffer_var.value[current_idx] = input_var.value
