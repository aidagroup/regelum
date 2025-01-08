import numpy as np
import pytest

from regelum.node.memory.lag import Lag
from regelum.node.classic_control.envs.continuous import Pendulum
from regelum.node.classic_control.controllers.mpc import MPCContinuous
from regelum.node.graph import Graph
from regelum.node.base import Node


@pytest.fixture(autouse=True)
def cleanup_node_instances():
    """Clean up Node instances after each test."""
    yield
    Node._instances.clear()


def test_lag_with_pendulum():
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
    lag = Lag(
        variable_names=["pendulum_1.state", "mpc_1.mpc_action"],
        lag_size=3,
        step_size=0.01,
    )

    # Create graph
    graph = Graph(
        [pendulum, mpc, lag],
        initialize_inner_time=True,
        debug=True,
    )
    graph.resolve(variables=graph.variables)

    # Run for a few steps to initialize
    for _ in range(5):
        graph.step()

    # Get lag variables
    state_lags = [lag.find_variable(f"pendulum_1.state@lag_{i}") for i in range(3)]
    action_lags = [lag.find_variable(f"mpc_1.mpc_action@lag_{i}") for i in range(3)]

    # Check if all lag variables exist
    assert all(var is not None for var in state_lags), "State lag variables not found"
    assert all(var is not None for var in action_lags), "Action lag variables not found"

    # Check shapes
    assert state_lags[0].value.shape == (
        2,
    ), f"Wrong state shape: {state_lags[0].value.shape}"
    assert action_lags[0].value.shape == (
        1,
    ), f"Wrong action shape: {action_lags[0].value.shape}"

    # Store current values
    old_states = [var.value.copy() for var in state_lags]
    old_actions = [var.value.copy() for var in action_lags]

    # Run one more step
    graph.step()

    # Check if values shifted correctly (lag_1 should get lag_0's old value, lag_2 should get lag_1's old value)
    for i in range(1, 3):
        assert np.allclose(
            state_lags[i].value, old_states[i - 1]
        ), f"State lag {i} not shifted correctly"
        assert np.allclose(
            action_lags[i].value, old_actions[i - 1]
        ), f"Action lag {i} not shifted correctly"

    # Check if lag_0 has the current input values
    current_state = pendulum.state.value
    current_action = mpc.action.value
    assert np.allclose(
        state_lags[0].value, current_state
    ), "Newest state lag doesn't match current state"
    assert np.allclose(
        action_lags[0].value, np.atleast_1d(current_action)
    ), "Newest action lag doesn't match current action"
