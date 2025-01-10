"""Example of double pendulum visualization."""

import numpy as np
from regelum.node.classic_control.envs.continuous import DoublePendulum
from regelum.node.visualization.pygame_renderer import DoublePendulumRenderer
from regelum.node.graph import Graph
from regelum.node.reset import ResetEachNSteps
from regelum.node.misc.reward import RewardTracker
from regelum.node.classic_control.controllers.mpc import MPCContinuous


class DoublePendulumReward(RewardTracker):
    """Reward function for double pendulum tracking upright position."""

    def objective_function(self, x: np.ndarray) -> float:
        # Penalize deviation from upright position and high velocities
        theta1, theta2, omega1, omega2 = x[0], x[1], x[2], x[3]
        return (
            (theta1 - np.pi) ** 2  # First pendulum upright
            + (theta2 - np.pi) ** 2  # Second pendulum upright
            + 0.1 * omega1**2  # Dampen first pendulum velocity
            + 0.1 * omega2**2  # Dampen second pendulum velocity
        )

    @property
    def name(self) -> str:
        return "double_pendulum_reward"


if __name__ == "__main__":
    # Create double pendulum with interesting initial state
    pendulum = DoublePendulum(
        m1=1.0,  # Mass of first pendulum
        m2=1.0,  # Mass of second pendulum
        l1=1.0,  # Length of first pendulum
        l2=1.0,  # Length of second pendulum
        initial_state=np.array(
            [np.pi / 2, np.pi / 4, 0.0, 0.0]
        ),  # Start at 90° and 45°
        step_size=0.01,
        control_signal_name="mpc_1.mpc_action",
    )

    reward = DoublePendulumReward(state_variable=pendulum.state)
    mpc_node = MPCContinuous(
        controlled_system=pendulum,
        controlled_state=pendulum.state,
        control_dimension=2,
        objective_function=reward.objective_function,
        control_bounds=(np.array([-400, -400]), np.array([400, 400])),
        prediction_horizon=35,
        prediction_method=MPCContinuous.PredictionMethod.EULER,
        step_size=0.03,
    )

    # Create visualization with motion trails
    viz = DoublePendulumRenderer(
        state_variable=pendulum.state,
        fps=60.0,
        window_size=(1000, 400),
        visible_history=1000,
        reward_variable=reward.reward,
        trail_length=30,
        record_video=True,
        video_path="./examples/gfx/double_pendulum_viz.avi",
    )

    # Reset every 1000 steps to see different behaviors
    reset_node = ResetEachNSteps(
        node_name_to_reset=pendulum.external_name, n_steps=1000
    )

    # Create and configure graph
    graph = Graph(
        [pendulum, viz, reset_node, reward, mpc_node],
        initialize_inner_time=True,
        states_to_log=[pendulum.state.full_name, mpc_node.action.full_name],
        debug=True,
    )

    graph.resolve(graph.variables)

    # Run simulation
    n_steps = 1000
    for _ in range(n_steps):
        graph.step()
