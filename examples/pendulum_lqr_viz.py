"""Example of pendulum control using LQR with visualization."""

import numpy as np
from regelum.node.classic_control.envs.continuous import Pendulum
from regelum.node.classic_control.controllers.lqr import LQRController
from regelum.node.visualization.pygame_renderer import PendulumRenderer
from regelum.node.graph import Graph
from regelum.node.reset import ResetEachNSteps
from regelum.node.misc.reward import RewardTracker


class RewardPendulum(RewardTracker):
    """Reward function for pendulum stabilization."""

    @property
    def name(self) -> str:
        return "reward_pendulum"

    def objective_function(self, x: np.ndarray) -> float:
        angle_error = x[0]
        return float(4 * angle_error**2 + x[1] ** 2)


pendulum = Pendulum(
    control_signal_name="lqr_1.action", initial_state=np.array([np.pi, 0.0])
)

reward_tracker = RewardPendulum(state_variable=pendulum.state)

A = np.array([[0, 1], [-3 * pendulum.gravity_acceleration / (2 * pendulum.length), 0]])
B = np.array([[0], [1 / pendulum.mass]])
Q = np.diag([10.0, 1.0])
R = np.array([[0.1]])

lqr = LQRController(
    controlled_state=pendulum.state,
    system_matrices=(A, B),
    cost_matrices=(Q, R),
    control_limits=(-10.0, 10.0),
    step_size=0.01,
)

viz = PendulumRenderer(
    state_variable=pendulum.state,
    fps=240.0,
    window_size=(1200, 400),
    visible_history=1000,
    reward_variable=reward_tracker.reward,
)

reset_node = ResetEachNSteps(
    node_name_to_reset=pendulum.external_name,
    n_steps=1000,
)

graph = Graph(
    [pendulum, lqr, viz, reset_node, reward_tracker],
    initialize_inner_time=True,
    states_to_log=[
        pendulum.state.full_name,
        lqr.action.full_name,
        reward_tracker.reward.full_name,
        "step_counter_1.counter",
    ],
    logger_cooldown=0,
)
graph.resolve(graph.variables)

n_steps = 5000
for _ in range(n_steps):
    graph.step()
