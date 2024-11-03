from regelum.environment.node.base import Node, State, Inputs, Graph, Clock
from regelum.environment.transistor import Transistor, CasADiTransistor, ScipyTransistor
from regelum.utils import rg
import numpy as np
import pytest


class SimpleNode(Node):
    state = State("simple", (2,), np.array([1.0, 2.0]))
    inputs = ["Clock"]

    def __init__(self):
        super().__init__(step_size=0.1)

    def compute_state_dynamics(self):
        return {"simple": self.state.data + 0.1}


class ODENode(Node):
    state = State("ode", (2,), np.array([0.0, 1.0]))
    inputs = ["Clock"]

    def __init__(self):
        super().__init__(step_size=0.1)

    def compute_state_dynamics(self):
        x = self.state.data
        return {"ode": rg.vstack([x[1], -x[0]])}  # Simple harmonic oscillator


class HierarchicalNode(Node):
    state = State(
        "hierarchical",
        _value=[
            State("pos", shape=(2,), _value=np.array([0.0, 0.0])),
            State("vel", shape=(2,), _value=np.array([1.0, 1.0])),
        ],
    )
    inputs = ["Clock"]

    def __init__(self):
        super().__init__(step_size=0.1)

    def compute_state_dynamics(self):
        return {
            "hierarchical/pos": self.state["hierarchical/pos"].data + 0.1,
            "hierarchical/vel": self.state["hierarchical/vel"].data,
        }


def test_state_initialization():
    state = State("test", shape=(2,), _value=np.array([1.0, 2.0]))
    assert state.name == "test"
    assert state.shape == (2,)
    assert np.array_equal(state.data, np.array([1.0, 2.0]))
    assert state.is_leaf


def test_hierarchical_state():
    state = State(
        "parent",
        _value=[
            State("child1", shape=(2,), _value=np.array([1.0, 2.0])),
            State("child2", shape=(1,), _value=np.array([3.0])),
        ],
    )
    assert not state.is_leaf
    assert len(state.paths) == 2
    assert "parent/child1" in state.paths
    assert "parent/child2" in state.paths


def test_inputs_resolution():
    node1 = SimpleNode()
    node2 = ODENode()
    graph = Graph([node1, node2])

    assert node1.inputs._resolved
    assert node2.inputs._resolved
    assert len(node1.inputs.states) == 1
    assert node1.inputs.states[0].name == "Clock"


def test_simple_transistor():
    node = SimpleNode()
    node.with_transistor(Transistor)

    initial_state = node.state.data.copy()
    node.transistor.step()

    assert np.array_equal(node.state.data, initial_state + 0.1)


@pytest.mark.parametrize("transistor_class", [CasADiTransistor, ScipyTransistor])
def test_ode_transistors(transistor_class):
    node = ODENode()

    # Create graph to resolve inputs
    graph = Graph([node])
    node.with_transistor(transistor_class, dynamic_variable_paths=["ode"])

    initial_state = node.state.data.copy()
    graph.step()

    # For harmonic oscillator, energy = 0.5 * (velocity^2 + position^2)
    # x[0] is position, x[1] is velocity
    energy_initial = 0.5 * (initial_state[1] ** 2 + initial_state[0] ** 2)
    energy_final = 0.5 * (node.state.data[1] ** 2 + node.state.data[0] ** 2)
    assert abs(energy_final - energy_initial) < 1e-9


def test_hierarchical_node():
    node = HierarchicalNode()
    node.with_transistor(Transistor)

    initial_pos = node.state["hierarchical/pos"].data.copy()
    initial_vel = node.state["hierarchical/vel"].data.copy()

    node.transistor.step()

    assert np.array_equal(node.state["hierarchical/pos"].data, initial_pos + 0.1)
    assert np.array_equal(node.state["hierarchical/vel"].data, initial_vel)


def test_graph_resolution():
    node1 = SimpleNode()
    node2 = ODENode()
    node3 = HierarchicalNode()

    graph = Graph([node1, node2, node3])

    # Test if all nodes have transistors after graph creation

    node1.with_transistor(Transistor)
    node2.with_transistor(CasADiTransistor, dynamic_variable_paths=["ode"])
    node3.with_transistor(Transistor)

    # Test graph step
    graph.step()

    # Clock should be present and incrementing
    clock_node = next(node for node in graph.nodes if isinstance(node, Clock))
    assert clock_node.state.data[0] == clock_node.fundamental_step_size


def test_state_path_search():
    node = HierarchicalNode()

    pos_state = node.state.search_by_path("hierarchical/pos")
    vel_state = node.state.search_by_path("hierarchical/vel")

    assert pos_state is not None
    assert vel_state is not None
    assert pos_state.name == "pos"
    assert vel_state.name == "vel"
    assert pos_state.shape == (2,)
    assert vel_state.shape == (2,)


def test_graph_with_logging():
    node = ODENode()
    graph = Graph([node], states_to_log=["ode"])
    node.with_transistor(CasADiTransistor, dynamic_variable_paths=["ode"])
    # Run a few steps
    for _ in range(5):
        graph.step()

    # Check if logger collected data
    assert len(graph.logger.logs["time"]) == 5
    assert len(graph.logger.logs["ode"]) == 5
    assert all(isinstance(log, np.ndarray) for log in graph.logger.logs["ode"])
