"""Pendulum MPC example."""

from regelum.node.classic_control.envs.continuous import Pendulum
from regelum.node.graph import Graph
from regelum.node.classic_control.controllers.mpc import MPCContinuous
from regelum.node.reset import ResetEachNSteps
import numpy as np


pendulum = Pendulum(
    control_signal_name="mpc_1.mpc_action",
    state_reset_modifier=lambda x: x + np.random.randn(2) * 0.1,
)
reset_pendulum = ResetEachNSteps(node_name_to_reset=pendulum.external_name, n_steps=10)
mpc_node = MPCContinuous(
    controlled_system=pendulum,
    controlled_state=pendulum.state,
    control_dimension=1,
    objective_function=pendulum.objective_function,
    control_bounds=(np.array([-10]), np.array([10])),
)

graph = Graph(
    [mpc_node, pendulum, reset_pendulum],
    initialize_inner_time=True,
    states_to_log=[
        pendulum.state.full_name,
        mpc_node.action.full_name,
        "step_counter_1.counter",
    ],
    logger_cooldown=0,
)
graph.resolve(graph.variables)

n_steps = 11

for _ in range(n_steps):
    graph.step()
