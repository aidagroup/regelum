from regelum.utils import rg
from regelum.environment.node.base import Node, State, Inputs
import numpy as np


class Pendulum(Node):
    state = State(
        "pendulum_state",
        (2,),
        np.array([np.pi, 0]),
        # _reset_modifier=lambda x: x + np.random.uniform(-0.5, 0.5, size=2),
    )
    inputs = Inputs(["action", "reset_pendulum_state"])
    length = 1
    mass = 1
    gravity_acceleration = 9.81

    def system_dynamics(self, x, u):
        pendulum_mpc_control = u

        angle = x[0]
        angular_velocity = x[1]
        torque = pendulum_mpc_control

        d_angle = angular_velocity
        d_angular_velocity = self.gravity_acceleration / self.length * rg.sin(
            angle
        ) + torque / (self.mass * self.length**2)

        return {"pendulum_state": rg.vstack([d_angle, d_angular_velocity])}

    def compute_state_dynamics(self):
        pendulum_mpc_control = self.inputs["action"].data

        return self.system_dynamics(self.state.data, pendulum_mpc_control)
