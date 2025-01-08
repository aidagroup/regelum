"""Test buffer functionality."""

import numpy as np
import pytest

from regelum.node.memory.buffer import DataBuffer
from regelum.node.classic_control.envs.continuous import Pendulum
from regelum.node.classic_control.controllers.mpc import MPCContinuous
from regelum.node.graph import Graph
from regelum.node.base import Node


@pytest.fixture(autouse=True)
def cleanup_node_instances():
    """Clean up Node instances after each test."""
    yield
    Node._instances.clear()


def test_buffer_with_pendulum():
    # Initialize nodes
    pendulum = Pendulum(
        control_signal_name="mpc_1.mpc_action",
    )
    mpc = MPCContinuous(
        controlled_system=pendulum,
        controlled_state=pendulum.state,
        control_dimension=1,
        objective_function=pendulum.objective_function,
        control_bounds=(np.array([-2.0]), np.array([2.0])),
    )
    buffer = DataBuffer(
        variable_names=["pendulum_1.state", "mpc_1.mpc_action"],
        buffer_size=10,
        step_size=0.01,
    )

    # Create graph
    graph = Graph(
        [pendulum, mpc, buffer],
        initialize_inner_time=True,
        debug=True,
    )
    graph.resolve(variables=graph.variables)

    # Run simulation for buffer_size steps
    for _ in range(10):
        graph.step()

    # Get buffers
    state_buffer = buffer.find_variable("pendulum_1.state@buffer")
    action_buffer = buffer.find_variable("mpc_1.mpc_action@buffer")

    assert state_buffer is not None, "State buffer not found"
    assert action_buffer is not None, "Action buffer not found"
    assert state_buffer.value is not None, "State buffer not initialized"
    assert action_buffer.value is not None, "Action buffer not initialized"

    # Check buffer shapes
    assert state_buffer.value.shape == (
        10,
        2,
    ), f"Wrong state buffer shape: {state_buffer.value.shape}"
    assert action_buffer.value.shape == (
        10,
    ), f"Wrong action buffer shape: {action_buffer.value.shape}"

    # Check if buffers contain non-zero values (simulation happened)
    assert not np.allclose(state_buffer.value, 0), "State buffer is all zeros"
    assert not np.allclose(action_buffer.value, 0), "Action buffer is all zeros"

    # Test circular buffer behavior
    old_state_buffer = state_buffer.value.copy()
    old_action_buffer = action_buffer.value.copy()

    # Run for 5 more steps
    for _ in range(5):
        graph.step()

    # Check that first 5 entries are different (overwritten)
    assert not np.allclose(
        old_state_buffer[:5], state_buffer.value[:5]
    ), "Circular buffer not working for state"
    assert not np.allclose(
        old_action_buffer[:5], action_buffer.value[:5]
    ), "Circular buffer not working for action"

    # Last 5 entries should be the same (not overwritten yet)
    assert np.allclose(
        old_state_buffer[5:], state_buffer.value[5:]
    ), "Buffer overwrote wrong entries for state"
    assert np.allclose(
        old_action_buffer[5:], action_buffer.value[5:]
    ), "Buffer overwrote wrong entries for action"
