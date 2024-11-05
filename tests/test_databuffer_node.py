import pytest
import numpy as np
from regelum.environment.node.base import Node, State, Graph
from regelum.environment.node.memory.data_buffer import DataBuffer


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


@pytest.fixture
def dynamic_node():
    node = DynamicNode()
    node.state["dynamic/pos"].data[:] = np.array([0.0, 0.0])
    node.state["dynamic/vel"].data[:] = np.array([1.0, 1.0])
    return node


@pytest.fixture
def data_buffer(dynamic_node):
    buffer = DataBuffer(
        dynamic_node,
        paths_to_remember=["dynamic/pos", "dynamic/vel"],
        buffer_size=5,
        step_size=0.1,
    )
    return buffer


@pytest.fixture
def graph(dynamic_node, data_buffer):
    return Graph([dynamic_node, data_buffer])


def test_buffer_initialization(data_buffer):
    assert data_buffer.state.name == "buffer"
    assert data_buffer.buffer_size == 5
    assert not data_buffer.is_buffer_full
    assert data_buffer.buffer_idx == 0

    # Check buffer shapes
    assert data_buffer.state["buffer/pos_buffer"].shape == (5, 2)
    assert data_buffer.state["buffer/vel_buffer"].shape == (5, 2)


def test_buffer_filling(graph, data_buffer):
    # Step 1
    graph.step()
    assert data_buffer.buffer_idx == 1
    assert not data_buffer.is_buffer_full
    assert np.allclose(
        data_buffer.state["buffer/pos_buffer"].data[0], np.array([0.1, 0.1])
    )
    assert np.allclose(
        data_buffer.state["buffer/vel_buffer"].data[0], np.array([1.0, 1.0])
    )

    # Fill buffer
    for _ in range(4):
        graph.step()

    assert data_buffer.buffer_idx == 0
    assert data_buffer.is_buffer_full
    assert np.allclose(
        data_buffer.state["buffer/pos_buffer"].data[-1], np.array([0.5, 0.5])
    )


def test_get_buffer_data(graph, data_buffer):
    # Partial fill
    graph.step()
    data = data_buffer.get_buffer_data()
    assert len(data["pos"]) == 1
    assert len(data["vel"]) == 1

    # Full buffer
    for _ in range(4):
        graph.step()

    data = data_buffer.get_buffer_data()
    assert len(data["pos"]) == 5
    assert len(data["vel"]) == 5
    assert np.allclose(data["pos"][-1], np.array([0.5, 0.5]))


def test_partial_paths_buffer(dynamic_node):
    buffer = DataBuffer(
        dynamic_node,
        paths_to_remember=["dynamic/pos"],
        buffer_size=3,
        step_size=0.1,
    )
    graph = Graph([dynamic_node, buffer])

    graph.step()
    assert "vel" not in buffer.get_buffer_data()
    assert np.allclose(buffer.get_buffer_data()["pos"][0], np.array([0.1, 0.1]))


def test_multi_node_buffer(dynamic_node):
    # Create a second node
    class SecondNode(Node):
        state = State(
            "second",
            None,
            [State("temp", shape=(1,), _value=np.array([20.0]))],
        )

        def __init__(self):
            super().__init__(step_size=0.1, is_root=True)

        def compute_state_dynamics(self):
            return {"second/temp": self.state["second/temp"].data + 0.1}

    second_node = SecondNode()

    # Create buffer with both nodes
    buffer = DataBuffer(
        nodes_to_buffer=[dynamic_node, second_node],
        paths_to_remember=["dynamic/pos", "second/temp"],
        buffer_size=3,
        step_size=0.1,
    )
    graph = Graph([dynamic_node, second_node, buffer])

    # Test data collection
    graph.step()
    data = buffer.get_buffer_data()
    assert "pos" in data
    assert "temp" in data
    assert np.allclose(data["pos"][0], np.array([0.1, 0.1]))
    assert np.allclose(data["temp"][0], np.array([20.1]))
