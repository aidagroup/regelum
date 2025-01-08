"""Three wheeled robot dynamic MPC example."""

from regelum.node.classic_control.envs.continuous import ThreeWheeledRobotDynamic
from regelum.node.graph import Graph
from regelum.node.classic_control.controllers.mpc import MPCContinuous
from regelum.node.reset import ResetEachNSteps
import numpy as np
from regelum.utils import rg


def objective_function(x):
    return rg.sum(x[:3] ** 2)  # Minimize distance from origin for position states


robot = ThreeWheeledRobotDynamic(
    control_signal_name="mpc_1.mpc_action",
    state_reset_modifier=lambda x: x + np.random.randn(5) * 0.1,
    initial_state=np.array([1.0, 1.0, 0.0, 0.0, 0.0]),
    mass=10.0,
    inertia=1.0,
)
reset_robot = ResetEachNSteps(node_name_to_reset=robot.external_name, n_steps=1100)
mpc_node = MPCContinuous(
    controlled_system=robot,
    controlled_state=robot.state,
    control_dimension=2,
    objective_function=objective_function,
    control_bounds=(
        np.array([-50.0, -10.0]),  # [Force, Momentum]
        np.array([50.0, 10.0]),
    ),
    prediction_horizon=10,
)

graph = Graph(
    [mpc_node, robot, reset_robot],
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
