from regelum.environment.node.nodes.classic_control.envs.continuous.pendulum import (
    Pendulum,
)
from regelum.environment.node.nodes.graph import Graph
from regelum.environment.node.nodes.classic_control.controllers.mpc import MPCContinuous
from regelum.environment.node.nodes.reset import Reset
import numpy as np


class PendulumReset(Reset):
    def __init__(self, reset_interval: int):
        super().__init__(name="reset_pendulum_1", inputs=["step_counter_1.counter"])
        self.reset_interval = reset_interval
        self.step_counter = 0

    def step(self) -> None:
        step_counter = self.resolved_inputs.find("step_counter_1.counter").value
        self.flag.value = step_counter % self.reset_interval == 0


reset_pendulum = PendulumReset(reset_interval=10)

pendulum = Pendulum(
    control_signal_name="mpc_1.mpc_action",
    state_reset_modifier=lambda x: x + np.random.randn(2) * 0.1,
)
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
