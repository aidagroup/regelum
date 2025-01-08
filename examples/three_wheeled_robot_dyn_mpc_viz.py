from regelum.node.classic_control.envs.continuous import ThreeWheeledRobotDynamic
from regelum.node.graph import Graph
from regelum.node.classic_control.controllers.mpc import MPCContinuous
from regelum.node.reset import ResetEachNSteps
from regelum.node.visualization.pygame_renderer import ThreeWheeledRobotRenderer
import numpy as np
from regelum.utils import rg
from regelum.node.misc.reward import RewardTracker
from regelum.utils import NumericArray


class RewardRobot(RewardTracker):
    """Reward function for the robot."""

    def objective_function(self, x: NumericArray) -> float:
        return rg.sum(x[:3] ** 2)


robot = ThreeWheeledRobotDynamic(
    control_signal_name="mpc_1.mpc_action",
    state_reset_modifier=lambda x: x + np.random.randn(5) * 0.1,
    initial_state=np.array([1.0, 1.0, 0.0, 0.0, 0.0]),
    mass=10.0,
    inertia=1.0,
)
reward_tracker = RewardRobot(state_variable=robot.state)
reset_robot = ResetEachNSteps(node_name_to_reset=robot.external_name, n_steps=1100)
mpc_node = MPCContinuous(
    controlled_system=robot,
    controlled_state=robot.state,
    control_dimension=2,
    objective_function=reward_tracker.objective_function,
    control_bounds=(
        np.array([-50.0, -10.0]),
        np.array([50.0, 10.0]),
    ),
    prediction_horizon=10,
)

viz = ThreeWheeledRobotRenderer(
    state_variable=robot.state,
    fps=60.0,
    window_size=(800, 400),
    visible_history=1000,
    reward_variable=reward_tracker.reward,
)

graph = Graph(
    [mpc_node, robot, reset_robot, viz, reward_tracker],
    initialize_inner_time=True,
    states_to_log=[
        robot.state.full_name,
        mpc_node.action.full_name,
        "step_counter_1.counter",
    ],
    logger_cooldown=0,
)
graph.resolve(graph.variables)

n_steps = 1100

for _ in range(n_steps):
    graph.step()
