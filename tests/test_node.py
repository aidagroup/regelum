import pytest
from regelum.environment.node import State, Inputs, Node, Graph, Clock, Terminate
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
