"""Example of DC motor visualization with MPC control."""

import numpy as np
from regelum.node.classic_control.envs.continuous import DCMotor
from regelum.node.visualization.pygame_renderer import DCMotorRenderer
from regelum.node.graph import Graph
from regelum.node.reset import ResetEachNSteps
from regelum.node.misc.reward import RewardTracker
from regelum.node.classic_control.controllers.mpc import MPCContinuous


class DCMotorReward(RewardTracker):
    """Reward function for DC motor position and speed control."""

    def objective_function(self, x: np.ndarray) -> float:
        # Track target position while minimizing speed and current
        theta, omega, current = x[0], x[1], x[2]
        target_pos = np.pi  # Target: 180 degrees

        return (
            5.0 * (theta - target_pos) ** 2  # Position tracking
            + 0.1 * omega**2  # Minimize speed
            + 0.01 * current**2  # Minimize current
        )

    @property
    def name(self) -> str:
        return "dc_motor_reward"


if __name__ == "__main__":
    # Create DC motor with custom parameters
    motor = DCMotor(
        control_signal_name="mpc_1.mpc_action",
        initial_state=np.array([0.0, 0.0, 0.0]),  # [theta, omega, current]
        params={
            "J": 0.01,  # Rotor inertia
            "b": 0.1,  # Viscous friction
            "K": 0.01,  # Motor constant
            "R": 1.0,  # Armature resistance
            "L": 0.5,  # Armature inductance
        },
    )

    # Create reward tracker
    reward = DCMotorReward(state_variable=motor.state)

    # Create MPC controller
    mpc = MPCContinuous(
        controlled_system=motor,
        controlled_state=motor.state,
        control_dimension=1,
        objective_function=reward.objective_function,
        control_bounds=(np.array([-12.0]), np.array([12.0])),  # Voltage limits
        prediction_horizon=20,
        step_size=0.01,
    )

    # Create visualization
    viz = DCMotorRenderer(
        state_variable=motor.state,
        fps=60.0,
        window_size=(1200, 600),
        visible_history=1000,
        reward_variable=reward.reward,
        show_current_flow=True,  # Enable current flow animation
    )

    # Reset every 1000 steps
    reset_node = ResetEachNSteps(
        node_name_to_reset=motor.external_name,
        n_steps=1000,
    )

    # Create and configure graph
    graph = Graph(
        [motor, viz, reset_node, reward, mpc],
        initialize_inner_time=True,
        states_to_log=[
            motor.state.full_name,
            mpc.action.full_name,
        ],
        debug=True,
    )

    graph.resolve(graph.variables)

    # Run simulation
    n_steps = 10000
    for _ in range(n_steps):
        graph.step()
