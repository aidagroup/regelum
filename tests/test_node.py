import pytest
from regelum.environment.node import State, Inputs, Node, Graph, Clock, Terminate
from regelum.environment.transistor import DiscreteTransistor
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
    return Inputs(["test/child1/child3", "test/child2"])


@pytest.fixture
def dummy_node(basic_state):
    return DummyNode(DiscreteTransistor(0.1, 10.0), state=basic_state)


def test_state_paths(nested_state):
    assert nested_state.paths == ["/parent/child1/child3", "/parent/child2"]


def test_state_search_by_path(nested_state):
    assert nested_state.search_by_path("parent/child1/child3").value == np.array([1.0])
    assert nested_state.search_by_path("parent/child2").value.tolist() == [2.0, 3.0]


def test_state_is_leaf(basic_state, nested_state):
    assert basic_state.is_leaf
    assert not nested_state.is_leaf


def test_inputs_resolve(basic_inputs, nested_state):
    basic_inputs.resolve([nested_state])
    assert len(basic_inputs.states) == 2
    assert basic_inputs.states[0].name == "child3"
    assert basic_inputs.states[1].name == "child2"


def test_graph_resolve():
    node1 = DummyNode(
        DiscreteTransistor(0.1, 10.0),
        state=State("node1", (1,), np.array([0.0])),
        is_root=True,
    )
    node2 = DummyNode(
        DiscreteTransistor(0.1, 10.0),
        state=State("node2", (1,), np.array([0.0])),
        inputs=["node1"],
    )
    node3 = DummyNode(
        DiscreteTransistor(0.1, 10.0),
        state=State("node3", (1,), np.array([0.0])),
        inputs=["node2"],
    )

    graph = Graph([node3, node1, node2])
    assert [node.state.name for node in graph.ordered_nodes] == [
        "node1",
        "node2",
        "node3",
    ]


def test_clock():
    node1 = DummyNode(
        DiscreteTransistor(0.1, 10.0), state=State("node1", (1,), np.array([0.0]))
    )
    node2 = DummyNode(
        DiscreteTransistor(0.2, 10.0), state=State("node2", (1,), np.array([0.0]))
    )

    clock = Clock([node1, node2])
    assert abs(clock.fundamental_step_size - 0.1) < 1e-9

    node3 = DummyNode(
        DiscreteTransistor(0.1, 10.0), state=State("node3", (1,), np.array([0.0]))
    )
    clock2 = Clock([node1, node3])
    assert abs(clock2.fundamental_step_size - 0.1) < 1e-9

    assert abs(clock.compute_state_dynamics({}) - 0.1) < 1e-9

    node4 = DummyNode(
        DiscreteTransistor(0.15, 10.0), state=State("node4", (1,), np.array([0.0]))
    )
    node5 = DummyNode(
        DiscreteTransistor(0.25, 10.0), state=State("node5", (1,), np.array([0.0]))
    )
    clock3 = Clock([node4, node5])
    assert abs(clock3.fundamental_step_size - 0.05) < 1e-9


def test_terminate(dummy_node):
    terminate = Terminate(dummy_node)
    assert terminate.state.name == "test_terminate"
    assert terminate.inputs.paths_to_states == ["Clock", "plant"]

    # Test termination condition
    assert not terminate.compute_state_dynamics({"Clock": np.array([5.0])})
    assert terminate.compute_state_dynamics({"Clock": np.array([15.0])})
