import pytest
import numpy as np
from regelum.node.base import Node
from regelum.node.graph import Graph
from regelum.node.reset import Reset
from regelum.utils import rg


@pytest.fixture(autouse=True)
def cleanup_node_instances():
    """Clean up Node instances after each test."""
    yield
    Node._instances.clear()


class PendulumPDController(Node):
    def __init__(self, kp: float = 0.01, kd: float = 0.01, step_size: float = 0.01):
        super().__init__(
            inputs=["pendulum_1.state", "clock_1.time"],
            step_size=step_size,
            is_continuous=False,
            name="pendulum_pd_control",
        )
        self.kp = kp
        self.kd = kd
        self.control = self.define_variable(
            "control", value=np.array([0.0]), shape=(1,)
        )

    def step(self) -> None:
        if self.resolved_inputs is None:
            return
        pendulum_state_var = self.resolved_inputs.find("pendulum_1.state")
        if pendulum_state_var is None or pendulum_state_var.value is None:
            return
        pendulum_state = pendulum_state_var.value
        angle = pendulum_state[0]
        angular_velocity = pendulum_state[1]
        self.control.value = np.array([-self.kp * angle - self.kd * angular_velocity])


class Pendulum(Node):
    def __init__(self):
        super().__init__(
            inputs=["pendulum_pd_control_1.control"],
            is_continuous=True,
            is_root=True,
            name="pendulum",
        )
        self.state = self.define_variable(
            "state", value=np.array([np.pi, 0]), shape=(2,)
        )
        self.length = 1
        self.mass = 1
        self.gravity_acceleration = 9.81

    def state_transition_map(self, x, u):
        angle, angular_velocity = x[0], x[1]
        torque = u[0]
        d_angle = angular_velocity
        d_angular_velocity = self.gravity_acceleration / self.length * rg.sin(
            angle
        ) + torque / (self.mass * self.length**2)
        return np.array([d_angle, d_angular_velocity])

    def step(self) -> None:
        if self.resolved_inputs is None:
            return
        control_var = self.resolved_inputs.find("pendulum_pd_control_1.control")
        if control_var is None or control_var.value is None:
            return
        self.state.value += self.step_size * self.state_transition_map(
            self.state.value, control_var.value
        )


class PendulumReset(Reset):
    def __init__(self, reset_interval: int, step_size: float = 0.01):
        super().__init__(name="reset_pendulum_1")
        self.reset_interval = reset_interval
        self.step_counter = 0

    def step(self) -> None:
        self.step_counter += 1
        self.flag.value = self.step_counter % self.reset_interval == 0


@pytest.fixture
def pendulum_system():
    pendulum = Pendulum()
    pd_controller = PendulumPDController(20, 20, step_size=0.01)
    reset_node = PendulumReset(reset_interval=10, step_size=0.01)

    graph = Graph(
        [pendulum, pd_controller, reset_node],
        states_to_log=[
            "pendulum_1.state",
            "pendulum_pd_control_1.control",
            "reset_pendulum_1_1.flag",
        ],
        initialize_inner_time=True,
        debug=True,
    )
    graph.resolve(variables=graph.variables)
    return graph, pd_controller, pendulum, reset_node


def test_reset_behavior(pendulum_system):
    graph, _, pendulum, reset_node = pendulum_system
    initial_pendulum_state = pendulum.state.value.copy()

    # Run until first reset
    for _ in range(reset_node.reset_interval + 1):
        graph.step()
        print(pendulum.state.value)

    # Check if pendulum state was reset to initial condition
    assert np.allclose(pendulum.state.value, initial_pendulum_state)
