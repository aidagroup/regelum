from regelum.environment.node import Node, State, Inputs, Graph
from regelum.environment.transistor import Transistor, CasADiTransistor
import numpy as np
from regelum.utils import rg
from logging import getLogger
import logging
import time

# Add this before creating your nodes
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class PendulumPDController(Node):
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


class Logger(Node):
    state = State("logger_state", (1,))
    inputs = Inputs(["pendulum_state", "pendulum_pd_control"])

    def __init__(self, log_wait_time: float = 0.0):
        super().__init__()
        self.logger = getLogger(__name__)
        # Set logging level to INFO to see all info messages
        self.logger.setLevel("INFO")
        self.log_wait_time = log_wait_time
        self._last_log_time = time.time()

    def compute_state_dynamics(self):
        pendulum_state = self.inputs["pendulum_state"].value["value"]
        pendulum_pd_control = self.inputs["pendulum_pd_control"].value["value"]

        # Only log every self.wait_time seconds
        current_time = time.time()
        if (current_time - self._last_log_time) >= self.log_wait_time:
            self.logger.info(f"Pendulum state: {pendulum_state}")
            self.logger.info(f"Pendulum pd control: {pendulum_pd_control}")
            self._last_log_time = current_time

        return {"logger_state": pendulum_state}


pd_controller = PendulumPDController()
pendulum = Pendulum(is_root=True)
logger = Logger(log_wait_time=0.05)
graph = Graph([pd_controller, pendulum, logger])


pd_controller.with_transistor(Transistor, step_size=0.01)
pendulum.with_transistor(CasADiTransistor, step_size=0.01)
logger.with_transistor(Transistor, step_size=0.01)

n_steps = 1000

for _ in range(n_steps):
    graph.step()
