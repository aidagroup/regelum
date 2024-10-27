from regelum.environment.node import Node, State, Inputs, Graph
from regelum.environment.transistor import Transistor, CasADiTransistor
import numpy as np
from regelum.utils import rg
from dataclasses import dataclass


class PendulumPDControl(Node):
    state = State("pendulum_pd_control", (1,))
    inputs = Inputs(["pendulum_state"])

    def __init__(self, kp: float = 10, kd: float = 1, is_root: bool = False):
        super().__init__(is_root)
        self.kp = kp
        self.kd = kd

    def compute_state_dynamics(self):
        pendulum_state = self.inputs["pendulum_state"].value["value"]

        angle = pendulum_state[0]
        angular_velocity = pendulum_state[1]

        return {"pendulum_pd_control": -self.kp * angle - self.kd * angular_velocity}


class Pendulum(Node):
    state = State("pendulum_state", (2,), np.array([np.pi, 0]))
    inputs = Inputs(["pendulum_pd_control"])
    length = 1
    mass = 1
    gravity_acceleration = 9.81

    def compute_state_dynamics(self):
        pendulum_pd_control = self.inputs["pendulum_pd_control"].value["value"]

        angle = self.state.value["value"][0]
        angular_velocity = self.state.value["value"][1]
        torque = pendulum_pd_control

        d_angle = angular_velocity
        d_angular_velocity = (
            -3 * self.gravity_acceleration / (2 * self.length) * rg.sin(angle)
            + torque / self.mass
        )

        return {"pendulum_state": rg.vstack([d_angle, d_angular_velocity])}


pd_controller = PendulumPDControl()
pendulum = Pendulum(is_root=True)
graph = Graph([pd_controller, pendulum])


pd_controller.with_transistor(Transistor, step_size=0.01)
pendulum.with_transistor(CasADiTransistor, step_size=0.01)


for _ in range(100):
    graph.step()
    print(pendulum.state.value["value"])
