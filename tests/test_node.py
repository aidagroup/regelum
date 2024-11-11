import pytest
from regelum.environment.node.base import State, Inputs, Node, Graph, Clock
from regelum.environment.transistor import Transistor
import numpy as np


class DummyNode(Node):
    def compute_state_dynamics(self, inputs):
        return np.array([1.0])


@pytest.fixture
def basic_state():
    return State("test", (1,), np.array([0.0]))


@pytest.fixture
def nested_state():
    return State(
        "parent",
        None,
        [
            State("child1", None, [State("child3", (1,), np.array([1.0]))]),
            State("child2", (2,), np.array([2.0, 3.0])),
        ],
    )


@pytest.fixture
def basic_inputs():
    return Inputs(["parent/child1/child3", "parent/child2"])


@pytest.fixture
def dummy_node(basic_state):
    return DummyNode(Transistor(0.1, 10.0), state=basic_state)


def test_state_paths(nested_state):
    assert nested_state.paths == ["parent/child1/child3", "parent/child2"]


def test_state_search_by_path(nested_state):
    assert nested_state.search_by_path("parent/child1/child3").data == np.array([1.0])
    assert nested_state.search_by_path("parent/child2").data.tolist() == [2.0, 3.0]


def test_state_is_leaf(basic_state, nested_state):
    assert basic_state.is_leaf
    assert not nested_state.is_leaf


def test_inputs_resolve(basic_inputs, nested_state):
    basic_inputs.resolve([nested_state])
    assert len(basic_inputs.states) == 2
    assert basic_inputs.states[0].name == "child3"
    assert basic_inputs.states[1].name == "child2"


def test_node_reset_with_paths(nested_state):
    # Create a test node with nested state
    class TestNode(Node):
        def compute_state_dynamics(self):
            return {}

    node = TestNode(state=nested_state)

    # Modify some values
    node.state["parent/child1/child3"].data = np.array([10.0])
    node.state["parent/child2"].data = np.array([20.0, 30.0])

    # Reset specific path
    found_states = node.reset(["parent/child1/child3"])

    # Check that only specified path was reset
    assert np.array_equal(node.state["parent/child1/child3"].data, np.array([1.0]))
    assert np.array_equal(node.state["parent/child2"].data, np.array([20.0, 30.0]))
    assert found_states == {"parent/child1/child3"}

    # Reset all states
    node.reset()

    # Check all states are reset to initial values
    assert np.array_equal(node.state["parent/child1/child3"].data, np.array([1.0]))
    assert np.array_equal(node.state["parent/child2"].data, np.array([2.0, 3.0]))


def test_node_reset_invalid_path(nested_state):
    class TestNode(Node):
        def compute_state_dynamics(self):
            return {}

    node = TestNode(state=nested_state)

    # Try to reset non-existent path
    found_states = node.reset(["invalid/path"])
    assert found_states == set()


def test_graph_reset(nested_state):
    # Create test nodes with nested states
    class TestNode1(Node):
        def compute_state_dynamics(self):
            return {}

    class TestNode2(Node):
        def compute_state_dynamics(self):
            return {}

    node1 = TestNode1(state=nested_state.with_altered_name("node1"), step_size=0.1)
    node2 = TestNode2(state=nested_state.with_altered_name("node2"), step_size=0.1)

    graph = Graph([node1, node2])

    # Modify values in both nodes
    node1.state["node1/child1/child3"].data = np.array([10.0])
    node1.state["node1/child2"].data = np.array([20.0, 30.0])
    node2.state["node2/child1/child3"].data = np.array([40.0])
    node2.state["node2/child2"].data = np.array([50.0, 60.0])

    # Test partial reset
    graph.reset(["node1/child1/child3", "node2/child2"])

    # Check that only specified paths were reset
    assert np.array_equal(node1.state["node1/child1/child3"].data, np.array([1.0]))
    assert np.array_equal(node1.state["node1/child2"].data, np.array([20.0, 30.0]))
    assert np.array_equal(node2.state["node2/child1/child3"].data, np.array([40.0]))
    assert np.array_equal(node2.state["node2/child2"].data, np.array([2.0, 3.0]))

    # Test full reset
    graph.reset()

    # Check all states are reset
    assert np.array_equal(node1.state["node1/child1/child3"].data, np.array([1.0]))
    assert np.array_equal(node1.state["node1/child2"].data, np.array([2.0, 3.0]))
    assert np.array_equal(node2.state["node2/child1/child3"].data, np.array([1.0]))
    assert np.array_equal(node2.state["node2/child2"].data, np.array([2.0, 3.0]))


def test_graph_reset_invalid_path(nested_state):
    class TestNode(Node):
        def compute_state_dynamics(self):
            return {}

    node = TestNode(state=nested_state, step_size=0.1)
    graph = Graph([node])

    # Test reset with invalid path
    with pytest.raises(ValueError, match="Could not find states: {'invalid/path'}"):
        graph.reset(["invalid/path"])
