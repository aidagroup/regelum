"""Example of cart-pole control using MPC."""

import numpy as np
from regelum.node.classic_control.envs.continuous import CartPole
from regelum.node.classic_control.controllers.mpc import MPCContinuous
from regelum.node.visualization.pygame_renderer import CartPoleRenderer
from regelum.node.graph import Graph
from regelum.node.reset import ResetEachNSteps
from regelum.node.misc.reward import RewardTracker


class RewardCartPole(RewardTracker):
    """Reward function for cart-pole stabilization."""

    @property
    def name(self) -> str:
        return "reward_cartpole"

    def objective_function(self, x: np.ndarray) -> float:
        angle_error = x[0]
        pos_error = x[1]
        return 4 * angle_error**2 + pos_error**2 + 0.1 * x[2] ** 2 + 0.1 * x[3] ** 2


# Create cart-pole system
cartpole = CartPole(
    control_signal_name="mpc_1.mpc_action",
    initial_state=np.array([np.pi, 0.0, 0.0, 0.0]),
)

# Create reward tracker
reward_tracker = RewardCartPole(state_variable=cartpole.state)

# Create MPC controller
mpc = MPCContinuous(
    controlled_system=cartpole,
    controlled_state=cartpole.state,
    control_dimension=1,
    objective_function=reward_tracker.objective_function,
    control_bounds=(np.array([-150.0]), np.array([150.0])),
    prediction_horizon=35,
    step_size=0.01,
)

# Create visualization
viz = CartPoleRenderer(
    state_variable=cartpole.state,
    fps=60.0,
    window_size=(1200, 400),
    visible_history=1000,
    reward_variable=reward_tracker.reward,
)

reset_node = ResetEachNSteps(
    node_name_to_reset=cartpole.external_name,
    n_steps=1000,
)

# Create and configure graph
graph = Graph(
    [cartpole, mpc, viz, reset_node, reward_tracker],
    initialize_inner_time=True,
    states_to_log=[
        cartpole.state.full_name,
        mpc.action.full_name,
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
