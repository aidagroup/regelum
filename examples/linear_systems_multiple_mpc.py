"""Example of linear systems with MPC control."""

import numpy as np
from regelum.node.classic_control.envs.continuous import MassSpringDamper, DCMotor
from regelum.node.classic_control.controllers.mpc import MPCContinuous
from regelum.node.graph import Graph
from regelum.node.reset import ResetEachNSteps
from regelum.node.misc.reward import RewardTracker


# Create systems
msd = MassSpringDamper(
    control_signal_name="mpc_1.mpc_action",
    initial_state=np.array([1.0, 0.0]),
)

motor = DCMotor(
    control_signal_name="mpc_2.mpc_action",
    initial_state=np.array([10.0, 10.0, 0.0]),
)


# Create reward functions
class MSDReward(RewardTracker):
    """Reward function for mass-spring-damper system."""

    def objective_function(self, x: np.ndarray) -> float:
        return x[0] ** 2 + 0.1 * x[1] ** 2

    @property
    def name(self) -> str:
        return "msd_reward"


class MotorReward(RewardTracker):
    """Reward function for DC motor system."""

    def objective_function(self, x: np.ndarray) -> float:
        return x[0] ** 2 + 0.1 * x[1] ** 2 + 0.01 * x[2] ** 2

    @property
    def name(self) -> str:
        return "motor_reward"


if __name__ == "__main__":
    reward_msd = MSDReward(state_variable=msd.state)
    reward_motor = MotorReward(state_variable=motor.state)

    # Create MPC controllers
    mpc_msd = MPCContinuous(
        controlled_system=msd,
        controlled_state=msd.state,
        control_dimension=1,
        objective_function=reward_msd.objective_function,
        control_bounds=(np.array([-10.0]), np.array([10.0])),
        prediction_horizon=20,
        step_size=0.01,
    )

    mpc_motor = MPCContinuous(
        controlled_system=motor,
        controlled_state=motor.state,
        control_dimension=1,
        objective_function=reward_motor.objective_function,
        control_bounds=(np.array([-12.0]), np.array([12.0])),
        prediction_horizon=20,
        step_size=0.01,
    )

    reset_msd = ResetEachNSteps(node_name_to_reset=msd.external_name, n_steps=1000)
    reset_motor = ResetEachNSteps(node_name_to_reset=motor.external_name, n_steps=1000)

    graph = Graph(
        [msd, mpc_msd, reset_msd, reward_msd]
        + [motor, mpc_motor, reset_motor, reward_motor],
        initialize_inner_time=True,
        states_to_log=[
            msd.state.full_name,
            mpc_msd.action.full_name,
            motor.state.full_name,
            mpc_motor.action.full_name,
        ],
        debug=True,
    )

    graph.resolve(graph.variables)

    n_steps = 10000
    for _ in range(n_steps):
        graph.step()
