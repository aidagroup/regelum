from regelum.environment.node.base import Node, State, Inputs, Graph
from regelum.environment.node.continuous import Pendulum
from typing import Dict, Any
import numpy as np

### Pendulum, Observer, IsTruncated, IsTerminated, PdController, RunningObjective, DataBuffer


class Observer(Node):
    def __init__(self, step_size: float = 0.1):
        state = State("observation", (3,), np.zeros(3))  # [x, y, velocity]
        inputs = ["pendulum_state"]
        super().__init__(step_size=step_size, state=state, inputs=inputs)

    def compute_state_dynamics(self) -> Dict[str, Any]:
        angle, velocity = self.inputs["pendulum_state"].data
        x = np.cos(angle)
        y = np.sin(angle)
        return {"observation": np.array([x, y, velocity])}


class IsTruncated(Node):
    def __init__(self, steps_to_truncate: int, **kwargs):
        state = State("is_truncated", (1,), False)
        inputs = ["step_counter"]
        self.steps_to_truncate = steps_to_truncate
        super().__init__(state=state, inputs=inputs, **kwargs)

    def compute_state_dynamics(self) -> Dict[str, Any]:
        residual = self.inputs["step_counter"].data[0] % self.steps_to_truncate
        return {self.state.name: residual == 0}


class IsTerminated(Node):
    def __init__(self, **kwargs):
        state = State("is_terminated", (1,), False)
        inputs = []
        super().__init__(state=state, inputs=inputs, **kwargs)

    def compute_state_dynamics(self) -> Dict[str, Any]:
        return {self.state.name: False}


class Reset(Node):
    def __init__(self, input_node: Node, **kwargs):
        state = State(f"reset_{input_node.state.name}", (1,), False)
        inputs = ["is_terminated", "is_truncated"]
        super().__init__(state=state, inputs=inputs, **kwargs)

    def compute_state_dynamics(self) -> Dict[str, Any]:
        return {
            self.state.name: self.inputs["is_terminated"].data
            or self.inputs["is_truncated"].data
        }


class PendulumPDController(Node):
    state = State("action", (1,))
    inputs = Inputs(["observation"])

    def __init__(self, kp: float = 10, kd: float = 10, step_size: float = 0.01):
        super().__init__(step_size=step_size)
        self.kp = kp
        self.kd = kd

    def compute_state_dynamics(self):
        observation = self.inputs["observation"].data

        cos_angle = observation[0]
        sin_angle = observation[1]
        velocity = observation[2]

        angle = np.arctan2(sin_angle, cos_angle)

        return {"action": -self.kp * angle - self.kd * velocity}


class RewardComputer(Node):
    def __init__(self, step_size: float = 0.1):
        state = State("reward", (1,), np.zeros(1))
        inputs = ["observation", "action"]
        super().__init__(step_size=step_size, state=state, inputs=inputs)

    def compute_state_dynamics(self) -> Dict[str, Any]:
        state = self.inputs["observation"].data
        action = self.inputs["action"].data

        cos_angle, sin_angle, angular_velocity = state
        angle = np.arctan2(sin_angle, cos_angle)
        position_cost = 1 * angle**2
        velocity_cost = 0.1 * angular_velocity**2
        control_cost = 0.01 * (action**2)
        reward = -(position_cost + velocity_cost + control_cost)

        return {"reward": np.array([reward])}


pendulum = Pendulum(is_root=True, is_continuous=True)
observer = Observer()
pd_controller = PendulumPDController(step_size=0.01)
is_truncated = IsTruncated(steps_to_truncate=100)
is_terminated = IsTerminated()
reset = Reset(input_node=pendulum)
reward_computer = RewardComputer()
graph = Graph(
    nodes=[
        pendulum,
        observer,
        pd_controller,
        is_truncated,
        is_terminated,
        reset,
        reward_computer,
    ],
    states_to_log=["pendulum_state", "observation", "action", "reward", "is_truncated"],
)

for _ in range(1000):
    graph.step()
