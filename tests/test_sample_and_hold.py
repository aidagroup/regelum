import pytest
import numpy as np
from regelum.environment.node.base import Node, State
from regelum.utils import rg
from regelum.environment.graph import Graph


class PendulumPDController(Node):
    state = State("pendulum_pd_control", (1,), np.array([0.0]))
    inputs = ["pendulum_state", "Clock"]

    def __init__(self, kp: float = 0.01, kd: float = 0.01, step_size: float = 0.01):
        super().__init__(step_size=step_size)
        self.kp = kp
        self.kd = kd

    def compute_state_dynamics(self):
        pendulum_state = self.inputs["pendulum_state"].data
        angle = pendulum_state[0]
        angular_velocity = pendulum_state[1]
        return {
            "pendulum_pd_control": np.array(
                [-self.kp * angle - self.kd * angular_velocity]
            )
        }


class Pendulum(Node):
    state = State("pendulum_state", (2,), np.array([np.pi, 0]))
    inputs = ["pendulum_pd_control", "reset_pendulum_state"]
    length = 1
    mass = 1
    gravity_acceleration = 9.81

    def system_dynamics(self, x, u):
        angle, angular_velocity = x[0], x[1]
        torque = u
        d_angle = angular_velocity
        d_angular_velocity = self.gravity_acceleration / self.length * rg.sin(
            angle
        ) + torque / (self.mass * self.length**2)
        return {"pendulum_state": rg.vstack([d_angle, d_angular_velocity])}

    def compute_state_dynamics(self):
        return self.system_dynamics(
            self.state.data, self.inputs["pendulum_pd_control"].data
        )


class TerminateSignal(Node):
    state = State("reset_pendulum_state", (1,), np.array([False]))

    def __init__(self, reset_interval: int, step_size: float = 0.01):
        super().__init__(step_size=step_size)
        self.reset_interval = reset_interval
        self.step_counter = 0

    def compute_state_dynamics(self):
        self.step_counter += 1
        should_terminate = self.step_counter % self.reset_interval == 0
        return {"reset_pendulum_state": np.array([should_terminate])}


@pytest.fixture
def pendulum_system():
    pd_controller = PendulumPDController(20, 20, step_size=0.01)
    pendulum = Pendulum(is_root=True, is_continuous=True)
    terminate_signal = TerminateSignal(reset_interval=10, step_size=0.001)

    graph = Graph(
        [pd_controller, pendulum, terminate_signal],
        states_to_log=[
            "pendulum_state",
            "pendulum_pd_control",
            "reset_pendulum_state",
        ],
    )
    return graph, pd_controller, pendulum, terminate_signal


def test_sample_and_hold_behavior(pendulum_system):
    graph, pd_controller, _, _ = pendulum_system

    # Run for a few steps and check if control stays constant between updates
    previous_control = None
    constant_count = 0

    for _ in range(5):
        graph.step()
        current_control = pd_controller.state.data[0]

        if previous_control is not None and np.allclose(
            current_control, previous_control
        ):
            constant_count += 1

        previous_control = current_control

    # Control should stay constant for some steps due to sample and hold
    assert constant_count > 0
