"""Tests for reward functionality."""

import numpy as np
import pytest
from regelum.node.base import Node
from regelum.node.misc.reward import RewardTracker
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
            value=np.array([1.0, 2.0]),
            metadata={"shape": (2,)},
        )

    def step(self) -> None:
        self.state.value += 1.0


class QuadraticReward(RewardTracker):
    """Simple quadratic reward tracker for testing."""

    @property
    def name(self) -> str:
        return "quadratic_reward"

    def objective_function(self, x: np.ndarray) -> float:
        return -float(np.sum(x * x))  # Negative quadratic reward


def test_basic_reward_tracking():
    """Test basic reward tracking functionality."""
    dummy = DummyNode()
    reward = QuadraticReward(state_variable=dummy.state)

    graph = Graph([dummy, reward], initialize_inner_time=True)
    graph.resolve(variables=graph.variables)
    graph.step()

    # Check initial reward
    expected_reward = -(2.0**2 + 3.0**2)
    print((reward.reward.value, expected_reward))
    assert np.isclose(reward.reward.value, expected_reward)

    # Step and check updated reward
    graph.step()
    expected_reward = -(3.0**2 + 4.0**2)
    assert np.isclose(reward.reward.value, expected_reward)


def test_reward_with_state_reset():
    """Test reward tracking with state resets."""
    dummy = DummyNode()
    reward = QuadraticReward(state_variable=dummy.state)

    graph = Graph([dummy, reward], initialize_inner_time=True)
    graph.resolve(variables=graph.variables)

    # Run a few steps
    for _ in range(3):
        graph.step()

    # Reset state and check reward updates
    dummy.state.reset()
    graph.step()
    expected_reward = -(2.0**2 + 3.0**2)
    assert np.isclose(reward.reward.value, expected_reward)


class TestReward:
    """Test suite for reward functionality."""

    def __init__(self):
        """Initialize test environment."""
