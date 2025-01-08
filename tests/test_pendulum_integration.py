"""Test pendulum integration."""

import pytest
import numpy as np
from typing import cast
from regelum.node.base import Node
from regelum.node.graph import Graph
from regelum.utils import rg


@pytest.fixture(autouse=True)
def reset_node_instances():
    Node._instances = {}
    yield


class Pendulum(Node):
    def __init__(self, step_size: float = 0.001) -> None:
        super().__init__(
            inputs=["controller_1.action"],
            step_size=step_size,
            is_continuous=True,
            is_root=True,
            name="pendulum",
        )
        self.length = 1.0
        self.mass = 1.0
        self.gravity_acceleration = 9.81
        self.state = self.define_variable(
            "state",
            value=np.array([np.pi, 0.0]),
            metadata={
                "shape": (2,),
                "reset_modifier": lambda x: np.array(
                    [np.random.uniform(-np.pi, np.pi), 0.0]
                ),
            },
        )

    def state_transition_map(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        angle, angular_velocity = state
        torque = action[0]
        d_angle = angular_velocity
        d_angular_velocity = self.gravity_acceleration / self.length * rg.sin(
            angle
        ) + torque / (self.mass * self.length**2)
        return np.array([d_angle, d_angular_velocity])

    def step(self) -> None:
        if self.resolved_inputs is None:
            return
        action = self.resolved_inputs.inputs[0].value
        if action is None:
            return
        derivatives = self.state_transition_map(self.state.value, action)
        self.state.value += self.step_size * derivatives


class PendulumPDController(Node):
    def __init__(
        self, kp: float = 0.01, kd: float = 0.01, step_size: float = 0.01
    ) -> None:
        super().__init__(
            inputs=["pendulum_1.state"],
            step_size=step_size,
            is_continuous=False,
            name="controller",
        )
        self.kp = kp
        self.kd = kd
        self.action = self.define_variable(
            "action",
            value=np.array([0.0]),
            metadata={"shape": (1,)},
        )

    def step(self) -> None:
        if self.resolved_inputs is None:
            return
        pendulum_state = self.resolved_inputs.find("pendulum_1.state")
        if pendulum_state is None or pendulum_state.value is None:
            return
        angle = pendulum_state.value[0]
        angular_velocity = pendulum_state.value[1]
        self.action.value[0] = -self.kp * angle - self.kd * angular_velocity


class DataBuffer(Node):
    def __init__(self, buffer_size: int = 1000, step_size: float = 0.01) -> None:
        super().__init__(
            inputs=["pendulum_1.state", "step_counter_1.counter"],
            step_size=step_size,
            is_continuous=False,
            name="buffer",
        )
        self.buffer_size = buffer_size
        self.state_buffer = self.define_variable(
            "state_buffer",
            value=np.zeros((buffer_size, 2)),
            metadata={"shape": (buffer_size, 2)},
        )

    def step(self) -> None:
        if self.resolved_inputs is None:
            return
        counter_var = self.resolved_inputs.find("step_counter.counter")
        state_var = self.resolved_inputs.find("pendulum.state")
        if (
            counter_var is None
            or counter_var.value is None
            or state_var is None
            or state_var.value is None
        ):
            return
        current_buffer_idx = counter_var.value - 1
        self.state_buffer.value[current_buffer_idx] = state_var.value


def test_basic_pendulum_control():
    """Test basic pendulum control with PD controller."""
    pendulum = Pendulum(step_size=0.01)
    controller = PendulumPDController(kp=10, kd=10, step_size=0.01)

    graph = Graph(
        [pendulum, controller],
        initialize_inner_time=True,
        debug=True,
    )
    graph.resolve(variables=graph.variables)

    # Run for a few steps and check if pendulum is being controlled
    initial_state = pendulum.state.value.copy()
    for _ in range(10):
        graph.step()

    # Check if controller is moving pendulum towards equilibrium (0, 0)
    final_state = pendulum.state.value
    assert np.abs(final_state[0]) < np.abs(
        initial_state[0]
    ), "Pendulum angle should move towards equilibrium"


def test_multi_episode_reset():
    """Test pendulum reset behavior across multiple episodes."""
    pendulum = Pendulum(step_size=0.01)
    controller = PendulumPDController(kp=10, kd=10, step_size=0.01)
    data_buffer = DataBuffer(buffer_size=10)

    graph = Graph(
        [pendulum, controller, data_buffer],
        initialize_inner_time=True,
        debug=True,
    )
    graph.resolve(variables=graph.variables)

    # Run multiple episodes with fewer steps
    initial_states = []
    for _ in range(3):
        pendulum.reset()
        initial_states.append(pendulum.state.value.copy())
        for _ in range(3):
            graph.step()

    # Check if reset generates different initial states
    assert not np.allclose(
        initial_states[0], initial_states[1]
    ), "Reset should generate different initial states"
    assert not np.allclose(
        initial_states[1], initial_states[2]
    ), "Reset should generate different initial states"


def test_parallel_execution():
    """Test parallel execution of multiple pendulum instances."""
    pendulum = Pendulum(step_size=0.01)
    controller = PendulumPDController(kp=10, kd=10, step_size=0.01)
    data_buffer = DataBuffer(buffer_size=10)

    graph = Graph(
        [pendulum, controller, data_buffer],
        initialize_inner_time=True,
        debug=True,
    )

    # Create subgraph with step counter
    subgraph = cast(
        Graph,
        graph.extract_as_subgraph(
            ["pendulum_1", "controller_1", "buffer_1"],
            n_step_repeats=5,
        ),
    )

    # Reset initial subgraph
    for node in subgraph.nodes:
        for var in node.variables:
            var.reset(apply_reset_modifier=True)

    # Clone graph and check if clones are independent
    cloned = cast(Graph, graph.clone_node("graph_2"))
    for node in cloned.nodes:
        for var in node.variables:
            var.reset(apply_reset_modifier=True)

    # Convert to parallel and run
    parallel_graph = graph.parallelize(n_workers=2)
    parallel_graph.step()
    # Check if both instances have different trajectories
    original_buffer = None
    cloned_buffer = None
    for node in parallel_graph.nodes:
        if isinstance(node, Graph):
            for subnode in node.nodes:
                if isinstance(subnode, DataBuffer):
                    if "buffer_1" in subnode.external_name:
                        original_buffer = subnode.state_buffer.value
                    elif "buffer_2" in subnode.external_name:
                        cloned_buffer = subnode.state_buffer.value

    assert (
        original_buffer is not None and cloned_buffer is not None
    ), "Both buffers should exist"
    assert not np.allclose(
        original_buffer, cloned_buffer
    ), "Parallel instances should have different trajectories"
