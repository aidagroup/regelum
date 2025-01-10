"""Buffer for storing data."""

from typing import List, Optional
import numpy as np

from regelum import Node, Inputs, Graph


class SingleBuffer(Node):
    """Buffer for storing single variable data."""

    def __init__(
        self,
        variable_full_name: str,
        buffer_size: Optional[int] = None,
        step_size: Optional[float] = None,
    ) -> None:
        """Initialize single variable buffer."""
        if buffer_size is None:
            buffer_size = 100
        if step_size is None:
            step_size = 0.01
        super().__init__(
            inputs=[variable_full_name, "step_counter_1.counter"],
            step_size=step_size,
            is_continuous=False,
            name="buffer",
        )
        self.buffer_size = buffer_size
        self.buffer = self.define_variable(
            f"buffer[{variable_full_name}]",
            value=None,
        )

    @property
    def inputs(self) -> Inputs:
        return self._inputs

    @inputs.setter
    def inputs(self, new_inputs: Inputs) -> None:
        """Set new inputs and update buffer variable name accordingly."""
        assert len(new_inputs.inputs) == len(
            self._inputs.inputs
        ), "New inputs must have the same length as the old inputs"
        old_var_name = f"buffer[{self._inputs.inputs[0]}]"
        new_var_name = f"buffer[{new_inputs.inputs[0]}]"
        self._inputs = new_inputs
        self.alter_variable_names({old_var_name: new_var_name})

    def step(self) -> None:
        if self.resolved_inputs is None:
            return

        input_var = self.resolved_inputs.find(self.inputs.inputs[0])
        if input_var is None or input_var.value is None:
            return

        # Initialize buffer if not done yet.
        # Done in order to avoid overwriting the buffer with the wrong shape.
        if self.buffer.value is None:
            shape = (self.buffer_size,) + np.array(input_var.value).shape
            self.buffer.value = np.zeros(shape)

        # Get current buffer index using modulo for circular buffer
        current_idx = (
            int(self.resolved_inputs.find("counter").value - 1) % self.buffer_size
        )

        self.buffer.value[current_idx] = input_var.value


class DataBuffer(Graph):
    """Buffer for storing multiple variable data."""

    def __init__(
        self,
        variable_full_names: List[str],
        buffer_sizes: Optional[List[int]] = None,
        step_sizes: Optional[List[float]] = None,
    ) -> None:
        """Initialize data buffer graph."""
        if buffer_sizes is None:
            buffer_sizes = [100] * len(variable_full_names)
        if step_sizes is None:
            step_sizes = [0.01] * len(variable_full_names)
        buffer_nodes = [
            SingleBuffer(var_full_name, buffer_size, step_size)
            for var_full_name, buffer_size, step_size in zip(
                variable_full_names, buffer_sizes, step_sizes
            )
        ]
        super().__init__(
            nodes=buffer_nodes,
            name="data_buffer",
        )
