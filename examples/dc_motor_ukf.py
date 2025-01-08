"""DC Motor state estimation using Unscented Kalman Filter.

This example demonstrates state estimation and control of a DC motor using:
1. Unscented Kalman Filter (UKF) for state estimation
2. Model Predictive Control (MPC) for position tracking
3. Partial noisy measurements (only position and current)

The system setup consists of:
- DC Motor: 3-state system [θ, ω, i] (angle, angular velocity, current)
- Measurements: Noisy observations of angle and current only
- Control: MPC using estimated state for feedback
- Visualization: Real-time display of true and estimated states

Key features:
- State estimation with partial observations
- Nonlinear state estimation using UKF
- MPC control using estimated states
- Real-time visualization of true vs estimated states

The example shows how to:
1. Set up a DC motor with realistic parameters
2. Create noisy partial measurements
3. Configure UKF for state estimation
4. Implement MPC using estimated states
5. Visualize true and estimated states in real-time
"""

import numpy as np
from regelum.node.classic_control.envs.continuous import DCMotor
from regelum.node.visualization.pygame_renderer import DCMotorRenderer
from regelum.node.graph import Graph
from regelum.node.misc.output import OutputPartial, OutputWithNoise
from regelum.node.classic_control.observers.unscented_kalman_filter import (
    UnscentedKalmanFilter,
)
from regelum.node.classic_control.controllers.mpc import MPCContinuous
from regelum.node.misc.reward import RewardTracker


class DCMotorReward(RewardTracker):
    """Reward function for DC motor position and speed control.

    Implements a quadratic reward function that penalizes:
    1. Deviation from target position (π radians)
    2. High angular velocities
    3. High currents

    The reward weights are:
    - Position error: 5.0
    - Angular velocity: 0.1
    - Current: 0.01
    """

    def objective_function(self, x: np.ndarray) -> float:
        """Compute the objective (negative reward) for the current state.

        Args:
            x: State vector [θ, ω, i] containing:
               - θ: Angular position [rad]
               - ω: Angular velocity [rad/s]
               - i: Current [A]

        Returns:
            float: Objective value (lower is better)
        """
        theta, omega, current = x[0], x[1], x[2]
        target_pos = np.pi  # Target: 180 degrees

        return (
            5.0 * (theta - target_pos) ** 2  # Position tracking
            + 0.1 * omega**2  # Minimize speed
            + 0.01 * current**2  # Minimize current
        )

    @property
    def name(self) -> str:
        """Get the name of this reward function.

        Returns:
            str: Name of the reward function
        """
        return "dc_motor_reward"


if __name__ == "__main__":
    # Create DC motor with custom parameters
    motor = DCMotor(
        control_signal_name="mpc_1.mpc_action",
        initial_state=np.array([0.0, 0.0, 0.0]),  # [theta, omega, current]
        params={
            "J": 0.01,  # Rotor inertia [kg⋅m²]
            "b": 0.1,  # Viscous friction [N⋅m⋅s/rad]
            "K": 0.01,  # Motor constant [N⋅m/A]
            "R": 1.0,  # Armature resistance [Ω]
            "L": 0.5,  # Armature inductance [H]
        },
    )

    # Create noisy measurements (σ = 0.1) of only position and current
    noisy_output = OutputWithNoise(
        observing_variable=motor.state,
        noise_std=0.1,  # Standard deviation of measurement noise
    )
    partial_output = OutputPartial(
        observing_variable=noisy_output.observed_value,
        observed_indices=[0, 2],  # Only observe θ and i
    )

    # Configure UKF for state estimation
    ukf = UnscentedKalmanFilter(
        system_node=motor,
        measurement_node=partial_output,
        initial_state=np.array([0.0, 0.0, 0.0]),
        initial_covariance=np.diag([0.1, 0.1, 0.1]),  # Initial uncertainty
        process_noise_cov=np.diag([0.01, 0.01, 0.01]),  # Process noise
        measurement_noise_cov=0.1 * np.eye(2),  # Measurement noise
        alpha=0.1,  # UKF tuning parameters
        beta=2.0,  # Optimal for Gaussian noise
        kappa=0.0,  # Secondary scaling parameter
        step_size=0.01,  # Integration time step [s]
    )

    # Create reward function using estimated state
    reward = DCMotorReward(state_variable=ukf.state_estimate)

    # Configure MPC controller
    mpc = MPCContinuous(
        controlled_system=motor,
        controlled_state=ukf.state_estimate,  # Use estimated state for control
        control_dimension=1,
        objective_function=reward.objective_function,
        control_bounds=(np.array([-12.0]), np.array([12.0])),  # Voltage limits [V]
        prediction_horizon=20,
        step_size=0.01,
    )

    # Create visualization
    viz = DCMotorRenderer(
        state_variable=motor.state,
        fps=60.0,
        window_size=(800, 400),
        visible_history=1000,
        reward_variable=reward.reward,
        show_current_flow=True,
        estimated_state_variable=ukf.state_estimate,  # Show estimated state
        record_video=True,
        video_path="./examples/gfx/dc_motor_ukf.avi",
    )

    # Configure computation graph
    graph = Graph(
        [motor, noisy_output, partial_output, ukf, mpc, reward, viz],
        initialize_inner_time=True,
        states_to_log=[
            motor.state.full_name,
            ukf.state_estimate.full_name,
            mpc.action.full_name,
            partial_output.observed_value.full_name,
        ],
        debug=True,
    )

    # Initialize graph
    graph.resolve(graph.variables)

    # Run simulation
    n_steps = 400
    for _ in range(n_steps):
        graph.step()
