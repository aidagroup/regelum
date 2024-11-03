from regelum.environment.node.base import Node, State, Inputs, Graph
import numpy as np
from regelum.utils import rg
import logging

# Add this before creating your nodes
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class PendulumPDController(Node):
    state = State("pendulum_pd_control", (1,))
    inputs = Inputs(["pendulum_state"])

    def __init__(self, kp: float = 0.01, kd: float = 0.01, step_size: float = 0.01):
        super().__init__(step_size=step_size)
        self.kp = kp
        self.kd = kd

    def compute_state_dynamics(self):
        pendulum_state = self.inputs["pendulum_state"].data

        angle = pendulum_state[0]
        angular_velocity = pendulum_state[1]

        return {"pendulum_pd_control": -self.kp * angle - self.kd * angular_velocity}


class Pendulum(Node):
    state = State("pendulum_state", (2,), np.array([np.pi, 0]))
    inputs = Inputs(["pendulum_pd_control"])
    length = 1
    mass = 1
    gravity_acceleration = 9.81

    def system_dynamics(self, x, u):
        pendulum_mpc_control = u

        angle = x[0]
        angular_velocity = x[1]
        torque = pendulum_mpc_control

        d_angle = angular_velocity
        d_angular_velocity = self.gravity_acceleration / (self.length) * rg.sin(
            angle
        ) + torque / (self.mass * self.length**2)

        return {"pendulum_state": rg.vstack([d_angle, d_angular_velocity])}

    def compute_state_dynamics(self):
        pendulum_mpc_control = self.inputs["pendulum_pd_control"].data

        return self.system_dynamics(self.state.data, pendulum_mpc_control)


pd_controller = PendulumPDController(20, 20, step_size=0.01)
pendulum = Pendulum(is_root=True, is_continuous=True)
graph = Graph(
    [pd_controller, pendulum], states_to_log=["pendulum_state", "pendulum_pd_control"]
)

n_steps = 100000

for _ in range(n_steps):
    graph.step()
