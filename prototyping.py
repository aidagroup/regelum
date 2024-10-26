from regelum.environment.node import Node, State, Inputs, Graph
from regelum.environment.transistor import CasADiTransistor, Transistor
from typing import Dict, Any
import numpy as np
from regelum.utils import rg


class Pendulum(Node):
    state = State("plant", (1, 2), np.array([np.pi, 0]))
    inputs = Inputs(["pendulum_control"])
    parameters = dict(m=1, l=1, g=9.81)

    def compute_state_dynamics(self) -> Dict[str, Any]:
        theta, theta_dot = self.state.value["value"][0], self.state.value["value"][1]
        m, length, g = self.parameters.values()
        u = self.inputs["pendulum_control"].value["value"]
        theta_ddot = (g / length) * np.sin(theta) + u / (m * length**2)

        return {"plant": rg.vstack([theta_dot, theta_ddot])}


class Controller(Node):
    state = State("pendulum_control", (1, 1))
    inputs = Inputs(["plant"])
    parameters = dict(Kp=15.0, Kd=8.0, setpoint=0)

    def compute_state_dynamics(self, inputs: Dict[str, Any]) -> Dict[str, Any]:

        theta, theta_dot = inputs["plant"]["value"][0], inputs["plant"]["value"][1]
        Kp, Kd, setpoint = self.parameters.values()

        error = setpoint - theta
        control = Kp * error - Kd * theta_dot

        return {"pendulum_control": np.array([control])}


pendulum_node = Pendulum(is_root=True)
controller_node = Controller()
graph = Graph([pendulum_node, controller_node])
pendulum_node.with_transistor(CasADiTransistor, step_size=0.01)
controller_node.with_transistor(Transistor, step_size=0.01)

for _ in range(200):
    graph.step()
    print(
        pendulum_node.state.value["value"][0],
        controller_node.state.value["value"],
    )
