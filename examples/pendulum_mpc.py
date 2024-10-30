from regelum.environment.node import Node, State, Inputs, Graph, MPCNode
from typing import Tuple, Optional, Dict, Any, Type
from regelum.environment.transistor import (
    Transistor,
    CasADiTransistor,
    ScipyTransistor,
    SampleAndHoldFactory,
)
import numpy as np
from regelum.utils import rg
import logging
import casadi as ca

# Add this before creating your nodes
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class Pendulum(Node):
    state = State("pendulum_state", (2,), np.array([np.pi, 0]))
    inputs = Inputs(["mpc_pendulum_state_control"])
    length = 1
    mass = 1
    gravity_acceleration = 9.81

    def system_dynamics(self, x, u):
        pendulum_mpc_control = u

        angle = x[0]
        angular_velocity = x[1]
        torque = pendulum_mpc_control

        d_angle = angular_velocity
        d_angular_velocity = (
            -3 * self.gravity_acceleration / (2 * self.length) * rg.sin(angle)
            + torque / self.mass
        )

        return {"pendulum_state": rg.vstack([d_angle, d_angular_velocity])}

    def compute_state_dynamics(self):
        pendulum_mpc_control = self.inputs["mpc_pendulum_state_control"].data

        return self.system_dynamics(self.state.data, pendulum_mpc_control)


class LoggerStepCounter(Node):
    state = State("step_counter", (1,), 0)
    inputs = Inputs(["Clock"])

    def __init__(self):
        super().__init__(is_root=False)

    def compute_state_dynamics(self):
        return {"step_counter": self.state.data + 1}


pendulum = Pendulum(is_root=True, step_size=0.01)
step_counter = LoggerStepCounter()
mpc_node = MPCNode(pendulum, control_shape=1, prediction_horizon=4)

graph = Graph(
    [mpc_node, pendulum, step_counter],
    states_to_log=["pendulum_state", "mpc_pendulum_state_control", "step_counter"],
    logger_cooldown=0.5,
)

zoh = SampleAndHoldFactory()
mpc_node.with_transistor(Transistor.with_modifier(zoh))
# pendulum.with_transistor(CasADiTransistor) # Uncomment to use CasADi
pendulum.with_transistor(ScipyTransistor)
step_counter.with_transistor(Transistor)
n_steps = 1000

for _ in range(n_steps):
    graph.step()
