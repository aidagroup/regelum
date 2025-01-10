"""Example of pendulum with friction control using energy-based controller."""

import numpy as np
from regelum.node.classic_control.envs.continuous import PendulumWithFriction
from regelum.node.classic_control.controllers.energy_based import (
    EnergyBasedSwingUpController,
)
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


# Create pendulum system with friction
pendulum = PendulumWithFriction(
    control_signal_name="energy_swing_up_1.action",
    initial_state=np.array([np.pi, 0.0]),  # Start from downward position
    friction_coeff=0.08,  # Quanser-like friction
)

# Create reward tracker
reward_tracker = RewardPendulum(state_variable=pendulum.state)

# Create energy-based controller with friction compensation
controller = EnergyBasedSwingUpController(
    controlled_state=pendulum.state,
    pendulum_params={
        "mass": pendulum.mass,
        "length": pendulum.length,
        "gravity": pendulum.gravity_acceleration,
        "friction": pendulum.friction_coeff,
    },
    control_limits=(-10.0, 10.0),
    gain=2.0,
    pd_gains=(10.0, 1.0),
    switch_threshold=0.95,
)

# Create visualization
viz = PendulumRenderer(
    state_variable=pendulum.state,
    fps=60.0,
    window_size=(1200, 400),
    visible_history=1000,
    reward_variable=reward_tracker.reward,
)

# Create reset node
reset_node = ResetEachNSteps(
    node_name_to_reset=pendulum.external_name,
    n_steps=1000,
)

# Create and configure graph
graph = Graph(
    [pendulum, controller, viz, reset_node, reward_tracker],
    initialize_inner_time=True,
    states_to_log=[
        pendulum.state.full_name,
        controller.action.full_name,
        reward_tracker.reward.full_name,
        "step_counter_1.counter",
    ],
    logger_cooldown=0,
)
graph.resolve(graph.variables)

# Run simulation
n_steps = 5000
for _ in range(n_steps):
    graph.step()
