"""Lag node for storing previous values."""

from typing import Dict, List
import numpy as np

from regelum.node.base import Node
from regelum import Variable


class Lag(Node):
    """Node that stores previous values of variables."""

    def __init__(
        self,
        variable_names: List[str],
        lag_size: int = 1,
        step_size: float = 0.01,
    ) -> None:
        """Initialize lag node.

        Args:
            variable_names: List of full variable names to track (format: "node_name.variable_name")
            lag_size: Number of previous values to store
            step_size: Time step for execution
        """
        assert variable_names is not None and lag_size > 0
        super().__init__(
            inputs=variable_names,
            step_size=step_size,
            is_continuous=False,
            name="lag",
        )
        self.lag_size = lag_size
        self._lags: Dict[str, List[Variable]] = {}

        for full_name in variable_names:
            self._lags[full_name] = [
                self.define_variable(
                    f"{full_name}@lag_{i}",
                    value=None,
                )
                for i in range(lag_size)
            ]

    def step(self) -> None:
        """Update lag variables with current values."""
        if self.resolved_inputs is None:
            return

        for full_name, lag_vars in self._lags.items():
            input_var = self.resolved_inputs.find(full_name)
            if input_var is None or input_var.value is None:
                continue

            # Convert input to numpy array if needed
            input_value = np.atleast_1d(np.array(input_var.value))

            # Initialize if not done yet
            if lag_vars[0].value is None:
                for lag_var in lag_vars:
                    lag_var.value = input_value.copy()
                    lag_var.metadata["shape"] = input_value.shape
                continue

            # Shift values
            for i in range(self.lag_size - 1, 0, -1):
                lag_vars[i].value = lag_vars[i - 1].value.copy()
            lag_vars[0].value = input_value
