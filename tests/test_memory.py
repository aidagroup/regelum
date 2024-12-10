import pytest
import numpy as np
from regelum.environment.node.base import Node, State
from regelum.environment.node.memory import MemoryCell, create_memory_chain
from regelum.environment.graph import Graph


class DynamicNode(Node):
    state = State(
        "dynamic",
        None,
        [
            State("pos", shape=(2,), _value=np.array([0.0, 0.0])),
            State("vel", shape=(2,), _value=np.array([1.0, 1.0])),
        ],
    )

    def __init__(self):
        super().__init__(step_size=0.1, is_root=True)

    def compute_state_dynamics(self):
        return {
            "dynamic/pos": self.state["dynamic/pos"].data
            + self.state["dynamic/vel"].data * 0.1,
            "dynamic/vel": self.state["dynamic/vel"].data,
        }


@pytest.fixture(autouse=True)
def reset_nodes():
    yield
    # Clean up any global state after each test


@pytest.fixture
def dynamic_node():
    node = DynamicNode()
    # Reset to initial values explicitly
    node.state["dynamic/pos"].data[:] = np.array([0.0, 0.0])
    node.state["dynamic/vel"].data[:] = np.array([1.0, 1.0])
    return node


@pytest.fixture
def memory_cell(dynamic_node):
    cell = MemoryCell(
        dynamic_node,
        paths_to_remember=["dynamic/pos", "dynamic/vel"],
        step_size=0.1,
    )
    return cell


@pytest.fixture
def graph(dynamic_node, memory_cell):
    return Graph([dynamic_node, memory_cell])


@pytest.fixture
def expected_positions():
    return {
        "t0": np.array([0.0, 0.0]),
        "t1": np.array([0.1, 0.1]),
        "t2": np.array([0.2, 0.2]),
        "t3": np.array([0.3, 0.3]),
    }


def test_memory_cell(graph, memory_cell, expected_positions):
    # Initial state check
    assert np.allclose(
        memory_cell.state["memory/current/pos"].data, expected_positions["t0"]
    )
    assert np.allclose(
        memory_cell.state["memory/current/vel"].data, np.array([1.0, 1.0])
    )

    graph.step()

    assert np.allclose(
        memory_cell.state["memory/previous/pos"].data, expected_positions["t0"]
    )
    assert np.allclose(
        memory_cell.state["memory/current/pos"].data, expected_positions["t1"]
    )


@pytest.fixture
def partial_memory_cell(dynamic_node):
    return MemoryCell(dynamic_node, paths_to_remember=["dynamic/pos"], step_size=0.1)


def test_memory_cell_partial_state(
    dynamic_node, partial_memory_cell, expected_positions
):
    graph = Graph([dynamic_node, partial_memory_cell])
    graph.step()

    assert np.allclose(
        partial_memory_cell.state["memory/previous/pos"].data, expected_positions["t0"]
    )
    assert np.allclose(
        partial_memory_cell.state["memory/current/pos"].data, expected_positions["t1"]
    )


@pytest.fixture
def memory_chain(dynamic_node):
    return create_memory_chain(
        target_node=dynamic_node,
        n_cells=3,
        paths_to_remember=["dynamic/pos"],
        step_size=0.1,
    )


def test_memory_chain(dynamic_node, memory_chain, expected_positions):
    graph = Graph([dynamic_node, *memory_chain])

    # Initial state check
    for i, cell in enumerate(memory_chain):
        assert np.allclose(
            cell.state[f"{i+1}_memory/current/pos"].data, expected_positions["t0"]
        )

    # Step 1
    graph.step()
    assert np.allclose(
        memory_chain[0].state[f"{1}_memory/current/pos"].data, expected_positions["t1"]
    )
    assert np.allclose(
        memory_chain[1].state[f"{2}_memory/current/pos"].data, expected_positions["t0"]
    )
    assert np.allclose(
        memory_chain[2].state[f"{3}_memory/current/pos"].data, expected_positions["t0"]
    )

    # Step 2
    graph.step()
    assert np.allclose(
        memory_chain[0].state[f"{1}_memory/current/pos"].data, expected_positions["t2"]
    )
    assert np.allclose(
        memory_chain[1].state[f"{2}_memory/current/pos"].data, expected_positions["t1"]
    )
    assert np.allclose(
        memory_chain[2].state[f"{3}_memory/current/pos"].data, expected_positions["t0"]
    )

    # Step 3
    graph.step()
    assert np.allclose(
        memory_chain[0].state[f"{1}_memory/current/pos"].data, expected_positions["t3"]
    )
    assert np.allclose(
        memory_chain[1].state[f"{2}_memory/current/pos"].data, expected_positions["t2"]
    )
    assert np.allclose(
        memory_chain[2].state[f"{3}_memory/current/pos"].data, expected_positions["t1"]
    )
