from regelum.environment.node.base_new import Node, Graph, Clock, Logger, StepCounter
import numpy as np
from numpy.typing import NDArray
import time
from statistics import mean, stdev
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from regelum.utils.logger import logger


class Pendulum(Node):
    """Pendulum system dynamics."""

    def __init__(self, step_size: float = 0.001) -> None:
        """Initialize Pendulum node.

        Args:
            step_size: Integration time step
        """
        super().__init__(
            inputs=["controller_1.action"],
            step_size=step_size,
            is_continuous=True,
            is_root=True,
            name="pendulum",
        )

        # Physical parameters
        self.length = 1.0
        self.mass = 1.0
        self.gravity_acceleration = 9.81

        # State variable
        self.state = self.define_variable(
            "state",
            value=np.array([np.pi, 0.0]),
            metadata={
                "shape": (2,),
            },
        )

    def state_transition_map(self, state: NDArray, action: NDArray) -> NDArray:
        """Compute state derivatives."""
        angle, angular_velocity = state
        torque = action[0]  # Assuming 1D control input

        d_angle = angular_velocity
        d_angular_velocity = self.gravity_acceleration / self.length * np.sin(
            angle
        ) + torque / (self.mass * self.length**2)

        return np.array([d_angle, d_angular_velocity])

    def step(self) -> None:
        """Update pendulum state using Euler integration."""
        # Heavier computational load
        # result = 0.0
        # for _ in range(50000):  # Much more iterations
        #     result += np.sin(np.random.random()) * np.cos(np.random.random())

        action = self.resolved_inputs.inputs[0].value
        derivatives = self.state_transition_map(self.state.value, action)
        self.state.value += self.step_size * derivatives


class PendulumPDController(Node):
    """PD controller for pendulum."""

    def __init__(
        self, kp: float = 0.01, kd: float = 0.01, step_size: float = 0.01
    ) -> None:
        """Initialize PD controller.

        Args:
            kp: Proportional gain
            kd: Derivative gain
            step_size: Control update interval
        """
        super().__init__(
            inputs=["pendulum_1.state"],
            step_size=step_size,
            is_continuous=False,
            name="controller",
        )

        self.kp = kp
        self.kd = kd

        # Control output
        self.action = self.define_variable(
            "action",
            value=np.array([0.0]),
            metadata={"shape": (1,)},
        )

    def step(self) -> None:
        """Compute control action using PD law."""
        # Add computational load
        # result = 0.0
        # for _ in range(50000):  # Match pendulum load
        #     result += np.sin(np.random.random()) * np.cos(np.random.random())

        pendulum_state = self.resolved_inputs.find("pendulum.state").value
        angle = pendulum_state[0]
        angular_velocity = pendulum_state[1]
        self.action.value[0] = -self.kp * angle - self.kd * angular_velocity


def create_pendulum_graph(debug: bool = False) -> tuple[Graph, list[Node]]:
    """Create a pendulum-controller graph."""
    # Create nodes
    pendulum = Pendulum(step_size=0.01)
    controller = PendulumPDController(kp=0.01, kd=0.01, step_size=0.01)
    logger = Logger(["pendulum_1.state"], step_size=0.01)
    clock = Clock(fundamental_step_size=0.01)
    step_counter = StepCounter([clock], start_count=0)
    nodes = [clock, step_counter, controller, pendulum, logger]

    return Graph(nodes, debug=debug), nodes


def main():
    pendulum = Pendulum(step_size=0.05)
    controller = PendulumPDController(kp=10, kd=10, step_size=0.01)
    logger = Logger(["pendulum_1.state"], step_size=0.01)
    clock = Clock(fundamental_step_size=0.01)
    step_counter = StepCounter([clock], start_count=0)
    nodes = [clock, step_counter, pendulum, logger]

    graph = Graph(nodes, debug=True)

    graph.insert_node(controller)

    graph.squash_into_subgraph(
        "step_counter_1 -> clock_1 -> pendulum_1 -> controller_1 -> logger_1",
        n_step_repeats=300,
    )
    graph.clone_node("graph_2")

    parallel_graph = graph.parallelize()
    parallel_graph.step()


if __name__ == "__main__":
    main()
