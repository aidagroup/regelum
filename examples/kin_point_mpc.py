from regelum.node.classic_control.envs.continuous.kin_point import KinematicPoint
from regelum.node.graph import Graph
from regelum.node.classic_control.controllers.mpc import MPCContinuous
from regelum.node.reset import ResetEachNSteps
import numpy as np
from regelum.utils import rg


def objective_function(x):
    return rg.sum(x**2)  # Minimize distance from origin


kin_point = KinematicPoint(
    control_signal_name="mpc_1.mpc_action",
    state_reset_modifier=lambda x: x + np.random.randn(2) * 0.1,
)
reset_kin_point = ResetEachNSteps(
    node_name_to_reset=kin_point.external_name, n_steps=10
)
mpc_node = MPCContinuous(
    controlled_system=kin_point,
    controlled_state=kin_point.state,
    control_dimension=2,
    objective_function=objective_function,
    control_bounds=(
        np.array([-10, -10]),  # .reshape(2, 1),
        np.array([10, 10]),  # reshape(2, 1),
    ),
)

graph = Graph(
    [mpc_node, kin_point, reset_kin_point],
    initialize_inner_time=True,
    states_to_log=[
        kin_point.state.full_name,
        mpc_node.action.full_name,
        "step_counter_1.counter",
    ],
    logger_cooldown=0,
)
graph.resolve(graph.variables)

n_steps = 11

for _ in range(n_steps):
    graph.step()
