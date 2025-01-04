from regelum.environment.node.nodes.base import Node
from regelum.environment.node.nodes.graph import Graph
from regelum.environment.node.core.inputs import Inputs
from regelum.environment.node.core.state import State
from regelum.environment.node.core.mpc_node_factory import MPCNodeFactory
import numpy as np
from regelum.utils import rg
import logging

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


class ThreeWheeledRobotKinematic(Node):
    state = State("robot_state", (3,), np.zeros(3))  # [x, y, angle]
    inputs = Inputs(["mpc_robot_state_control"])

    def __init__(self, is_root: bool = False, is_continuous: bool = True):
        super().__init__(is_root=is_root, is_continuous=is_continuous)
        self.action_bounds = {
            "mpc_robot_state_control": (
                np.array([-25.0, -5.0]),  # lower bounds
                np.array([25.0, 5.0]),  # upper bounds
            )
        }

    def system_dynamics(self, x: np.ndarray, u: np.ndarray):
        """Compute right-hand side of the dynamic system.

        Args:
            x: Current state [x, y, angle]
            u: Control inputs [velocity, angular_velocity]
        """
        Dstate = rg.zeros((3,), prototype=(x, u))

        # Kinematic model
        Dstate[0] = u[0] * rg.cos(x[2])  # dx/dt = v * cos(theta)
        Dstate[1] = u[0] * rg.sin(x[2])  # dy/dt = v * sin(theta)
        Dstate[2] = u[1]  # dtheta/dt = omega

        return {"robot_state": Dstate}

    def compute_state_dynamics(self):
        robot_control = self.inputs["mpc_robot_state_control"].data
        return self.system_dynamics(self.state.data, robot_control)


class LoggerStepCounter(Node):
    state = State("step_counter", (1,), 0)
    inputs = Inputs(["Clock"])

    def __init__(self):
        super().__init__(is_root=False)

    def compute_state_dynamics(self):
        return {"step_counter": self.state.data + 1}


pendulum = Pendulum(is_root=True, is_continuous=True)
robot = ThreeWheeledRobotKinematic(is_root=True, is_continuous=True)
step_counter = LoggerStepCounter()
mpc_pendulum_node = MPCNodeFactory(
    pendulum, control_shape=1, prediction_horizon=4, step_size=0.1
)
mpc_robot_node = MPCNodeFactory(
    robot, control_shape=2, prediction_horizon=4, step_size=0.1
)

graph = Graph(
    [mpc_pendulum_node, pendulum, step_counter, mpc_robot_node, robot],
    states_to_log=[
        "pendulum_state",
        "mpc_pendulum_state_control",
        "step_counter",
        "robot_state",
        "mpc_robot_state_control",
    ],
    logger_cooldown=0.5,
)

n_steps = 1000

for _ in range(n_steps):
    graph.step()
