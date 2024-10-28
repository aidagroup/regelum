from regelum.environment.node import Node, State, Inputs, Graph
from regelum.environment.transistor import Transistor, CasADiTransistor
import numpy as np
from regelum.utils import rg
from logging import getLogger
import logging
import time
import casadi as ca

# Add this before creating your nodes
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class PendulumMPCController(Node):
    state = State("pendulum_mpc_control", (1,))
    inputs = Inputs(["pendulum_state"])

    def __init__(self, prediction_horizon: int = 10, is_root: bool = False):
        super().__init__(is_root)
        # MPC parameters
        self.N = prediction_horizon  # Prediction horizon
        self.dt = 0.01  # Time step (should match transistor step size)

        # System parameters (matching Pendulum class)
        self.length = 1
        self.mass = 1
        self.g = 9.81

        # Initialize optimization problem
        self.opti = ca.Opti()

        # Decision variables
        self.X = self.opti.variable(2, self.N + 1)  # States [angle, angular_velocity]
        self.U = self.opti.variable(1, self.N)  # Control inputs [torque]

        # Parameters for current state
        self.x0 = self.opti.parameter(2, 1)

        # Define objective function
        objective = 0
        for k in range(self.N):
            # Penalize state deviation from upright position and control effort
            objective += (
                (self.X[0, k]) ** 2
                + 0.1 * (self.X[1, k]) ** 2
                + 0.01 * (self.U[0, k]) ** 2
            )

        self.opti.minimize(objective)

        # System dynamics constraints
        for k in range(self.N):
            x_next = self.X[:, k] + self.dt * self.system_dynamics(
                self.X[:, k], self.U[:, k]
            )
            self.opti.subject_to(self.X[:, k + 1] == x_next)

        # Input constraints
        self.opti.subject_to(self.opti.bounded(-5, self.U, 5))  # Limit torque

        # Initial condition constraints
        self.opti.subject_to(self.X[:, 0] == self.x0)

        # Set solver options
        opts = {"ipopt.print_level": 0, "print_time": 0}
        self.opti.solver("ipopt", opts)

    def system_dynamics(self, x, u):
        """Pendulum dynamics for MPC model"""
        angle, angular_velocity = x[0], x[1]
        torque = u[0]

        d_angle = angular_velocity
        d_angular_velocity = (
            -3 * self.g / (2 * self.length) * ca.sin(angle) + torque / self.mass
        )

        return ca.vertcat(d_angle, d_angular_velocity)

    def compute_state_dynamics(self):
        pendulum_state = self.inputs["pendulum_state"].value["value"]

        # Set current state parameter
        self.opti.set_value(self.x0, pendulum_state)

        try:
            # Solve optimization problem
            sol = self.opti.solve()
            # Extract first control input
            u_optimal = sol.value(self.U[:, 0])
        except:
            # Fallback control if optimization fails
            u_optimal = 0

        return {"pendulum_mpc_control": np.array([u_optimal])}


class Pendulum(Node):
    state = State("pendulum_state", (2,), np.array([np.pi, 0]))
    inputs = Inputs(["pendulum_mpc_control"])
    length = 1
    mass = 1
    gravity_acceleration = 9.81

    def compute_state_dynamics(self):
        pendulum_mpc_control = self.inputs["pendulum_mpc_control"].data

        angle = self.state.data[0]
        angular_velocity = self.state.data[1]
        torque = pendulum_mpc_control

        d_angle = angular_velocity
        d_angular_velocity = (
            -3 * self.gravity_acceleration / (2 * self.length) * rg.sin(angle)
            + torque / self.mass
        )

        return {"pendulum_state": rg.vstack([d_angle, d_angular_velocity])}


class Logger(Node):
    state = State("logger_state", (1,))
    inputs = Inputs(["pendulum_state", "pendulum_mpc_control"])

    def __init__(self, log_wait_time: float = 0.0):
        super().__init__()
        self.logger = getLogger(__name__)
        # Set logging level to INFO to see all info messages
        self.logger.setLevel("INFO")
        self.log_wait_time = log_wait_time
        self._last_log_time = time.time()

    def compute_state_dynamics(self):
        pendulum_state = self.inputs["pendulum_state"].data
        pendulum_mpc_control = self.inputs["pendulum_mpc_control"].data

        # Only log every self.wait_time seconds
        current_time = time.time()
        if (current_time - self._last_log_time) >= self.log_wait_time:
            self.logger.info(f"Pendulum state: {pendulum_state}")
            self.logger.info(f"Pendulum pd control: {pendulum_mpc_control}")
            self._last_log_time = current_time

        return {"logger_state": pendulum_state}


pd_controller = PendulumMPCController()
pendulum = Pendulum(is_root=True)
logger = Logger(log_wait_time=0.05)
graph = Graph([pd_controller, pendulum, logger])


pd_controller.with_transistor(Transistor, step_size=0.01)
pendulum.with_transistor(CasADiTransistor, step_size=0.01)
logger.with_transistor(Transistor, step_size=0.01)

n_steps = 1000

for _ in range(n_steps):
    graph.step()
