"""Output nodes for system observation and measurement.

This module provides nodes that emulate different types of system output measurements:
1. Basic output observation (Output)
2. Noisy measurements (OutputWithNoise)
3. Partial state observation (OutputPartial)

These nodes are useful for:
- Simulating real-world sensor measurements
- Testing state estimation algorithms
- Implementing output feedback control
- Evaluating robustness to measurement noise
- Creating partially observable scenarios

The module follows the Node architecture, where each output node:
- Observes a system variable
- Processes the observation (e.g., adds noise, selects components)
- Provides the processed observation through its observed_value Variable
"""

from regelum import Node
from regelum import Variable
from regelum.node.core.types import NumericArray
from typing import Optional, List
import numpy as np


class Output(Node):
    """Base output node for system observation.

    This node provides direct observation of a system variable without any
    modification. It serves as the base class for more specialized output
    nodes that implement various measurement effects.

    Attributes:
        observing_variable: Variable being observed
        observed_value: Variable containing the processed observation

    Example:
        ```python
        # Create direct output observer for a pendulum state
        output = Output(observing_variable=pendulum.state)
        # Access the observation
        observation = output.observed_value.value
        ```
    """

    def __init__(
        self, observing_variable: Variable, name: Optional[str] = None, suffix: str = ""
    ):
        """Initialize output node.

        Args:
            observing_variable: Variable to observe
            name: Optional name for the node
            suffix: Optional suffix for the observed variable
        """
        super().__init__(name=name, inputs=[observing_variable.full_name])
        self.observing_variable = observing_variable
        self.observed_value = self.define_variable(
            name=f"{observing_variable.name}@observed" + suffix,
            value=self.observe(self.observing_variable.value),
        )

    def observe(self, value: NumericArray) -> NumericArray:
        """Process the observed value.

        This base implementation returns the value unchanged.
        Derived classes should override this to implement specific
        measurement effects.

        Args:
            value: Raw value from the observed variable

        Returns:
            Processed observation (unchanged in base class)
        """
        return value

    def step(self) -> None:
        """Update the observation.

        Called each time step to update the observed_value with
        the latest measurement.
        """
        self.observed_value.value = self.observe(self.resolved_inputs.inputs[0].value)


class OutputWithNoise(Output):
    """Output node that adds Gaussian noise to measurements.

    Simulates noisy sensor measurements by adding zero-mean Gaussian noise
    with specified standard deviation to each component of the observed value.

    Attributes:
        noise_std: Standard deviation of the measurement noise

    Example:
        ```python
        # Create noisy observer for pendulum state
        noisy_output = OutputWithNoise(
            observing_variable=pendulum.state,
            noise_std=0.1  # Add noise with σ = 0.1
        )
        ```
    """

    def __init__(
        self, observing_variable: Variable, noise_std: float, name: Optional[str] = None
    ):
        """Initialize noisy output node.

        Args:
            observing_variable: Variable to observe
            noise_std: Standard deviation of the Gaussian noise
            name: Optional name for the node
        """
        self.noise_std = noise_std
        super().__init__(observing_variable, name, suffix="-noisy")

    def observe(self, value: NumericArray) -> NumericArray:
        """Add Gaussian noise to the observation.

        Args:
            value: Raw value from the observed variable

        Returns:
            Noisy observation: value + N(0, noise_std²)
        """
        return value + np.random.normal(0, self.noise_std, size=value.shape)


class OutputPartial(Output):
    """Output node that observes only selected components of a variable.

    Useful for simulating scenarios where only part of the state is measurable,
    such as:
    - Position-only measurements in mechanical systems
    - Partial state feedback control
    - Hidden state estimation problems

    Attributes:
        observed_indices: Indices of the components to observe

    Example:
        ```python
        # Create observer for only position in pendulum state [θ, ω]
        partial_output = OutputPartial(
            observing_variable=pendulum.state,
            observed_indices=[0]  # Only observe angle θ
        )
        ```
    """

    def __init__(
        self,
        observing_variable: Variable,
        observed_indices: List[int],
        name: Optional[str] = None,
    ):
        """Initialize partial output node.

        Args:
            observing_variable: Variable to observe
            observed_indices: List of indices to observe
            name: Optional name for the node
        """
        self.observed_indices = observed_indices
        super().__init__(observing_variable, name, suffix="-partial")

    def observe(self, value: NumericArray) -> NumericArray:
        """Select specified components of the observation.

        Args:
            value: Raw value from the observed variable

        Returns:
            Partial observation containing only the selected components
        """
        return value[self.observed_indices]
