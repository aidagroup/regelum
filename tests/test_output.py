"""Tests for output functionality."""

import numpy as np
import pytest
from regelum.node.base import Node
from regelum.node.misc.output import Output, OutputWithNoise, OutputPartial
from regelum.node.graph import Graph


@pytest.fixture(autouse=True)
def cleanup_node_instances():
    """Clean up Node instances after each test."""
    yield
    Node._instances.clear()


class DummyNode(Node):
    def __init__(self):
        super().__init__(name="dummy", inputs=[], step_size=0.1)
        self.state = self.define_variable(
            "state",
            value=np.array([1.0, 2.0, 3.0]),
            metadata={"shape": (3,)},
        )

    def step(self) -> None:
        self.state.value += 1.0


def test_basic_output():
    """Test basic output node functionality."""
    dummy = DummyNode()
    output = Output(observing_variable=dummy.state)

    graph = Graph([dummy, output], initialize_inner_time=True)
    graph.resolve(variables=graph.variables)

    # Check initial observation
    assert np.allclose(output.observed_value.value, [1.0, 2.0, 3.0])

    # Step and check updated observation
    graph.step()
    assert np.allclose(output.observed_value.value, [2.0, 3.0, 4.0])


def test_output_with_noise():
    """Test noisy output node functionality."""
    dummy = DummyNode()
    noise_std = 0.1
    noisy_output = OutputWithNoise(observing_variable=dummy.state, noise_std=noise_std)

    graph = Graph([dummy, noisy_output], initialize_inner_time=True)
    graph.resolve(variables=graph.variables)

    # Run multiple times to check noise properties
    n_samples = 1000
    noise_samples = np.zeros((n_samples, 3))

    for i in range(n_samples):
        graph.step()
        noise_samples[i] = noisy_output.observed_value.value - dummy.state.value

    # Check noise statistics
    mean_noise = np.mean(noise_samples, axis=0)
    std_noise = np.std(noise_samples, axis=0)

    # Mean should be close to 0 (within 3 sigma)
    assert np.all(np.abs(mean_noise) < 3 * noise_std / np.sqrt(n_samples))
    # Standard deviation should be close to noise_std
    assert np.all(np.abs(std_noise - noise_std) < 0.1)


def test_partial_output():
    """Test partial output node functionality."""
    dummy = DummyNode()
    indices = [0, 2]  # Only observe first and third components
    partial_output = OutputPartial(
        observing_variable=dummy.state, observed_indices=indices
    )

    graph = Graph([dummy, partial_output], initialize_inner_time=True)
    graph.resolve(variables=graph.variables)

    # Check initial partial observation
    assert np.allclose(partial_output.observed_value.value, [1.0, 3.0])

    # Step and check updated partial observation
    graph.step()
    assert np.allclose(partial_output.observed_value.value, [2.0, 4.0])


def test_output_chaining():
    """Test chaining multiple output nodes."""
    dummy = DummyNode()
    # Chain: Basic -> Noisy -> Partial
    basic_output = Output(observing_variable=dummy.state)
    noisy_output = OutputWithNoise(
        observing_variable=basic_output.observed_value, noise_std=0.1
    )
    partial_output = OutputPartial(
        observing_variable=noisy_output.observed_value, observed_indices=[0]
    )

    graph = Graph(
        [dummy, basic_output, noisy_output, partial_output], initialize_inner_time=True
    )
    graph.resolve(variables=graph.variables)

    # Check that the chain works
    graph.step()
    assert partial_output.observed_value.value.shape == (1,)
    assert isinstance(partial_output.observed_value.value, np.ndarray)


class TestOutput:
    """Test suite for output functionality."""

    def __init__(self):
        """Initialize test environment."""
