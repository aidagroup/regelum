"""Example of pendulum with motor control using backstepping.

This example demonstrates swing-up and stabilization of a pendulum with motor
dynamics using a hybrid backstepping controller that combines:
1. Energy-based swing-up with backstepping compensation
2. PD control near the upright position

The controller handles the motor dynamics explicitly through backstepping
during the swing-up phase, ensuring proper tracking of the energy-based
control signal.
"""

import numpy as np
from regelum.node.classic_control.envs.continuous.pendulum_motor import (
    PendulumWithMotor,
)
from regelum.node.classic_control.controllers.backstepping import PendulumBackstepping
from regelum.node.visualization.pygame_renderer import PendulumRenderer
from regelum.node.graph import Graph
from regelum.node.reset import ResetEachNSteps


if __name__ == "__main__":
    # Create pendulum with motor dynamics
    pendulum = PendulumWithMotor(
        control_signal_name="backstepping_1.action",
        initial_state=np.array([np.pi, 0.0, 0.0]),  # [angle, velocity, torque]
        motor_params={
            "mass": 0.1,  # kg
            "radius": 0.04,  # m
            "time_const": 0.05,  # s
        },
    )

    # Create backstepping controller
    controller = PendulumBackstepping(
        state_variable=pendulum.state,
        energy_gain=2.0,  # Gain for energy-based swing-up
        backstepping_gain=2.0,  # Gain for motor dynamics compensation
        switch_loc=0.1,  # Threshold for switching to PD control
        pd_coeffs=[10.0, 2.0],  # PD gains [kp, kd]
        action_min=-20.0,  # Minimum motor command
        action_max=20.0,  # Maximum motor command
        system=pendulum,
        step_size=0.01,
    )

    # Create visualization
    viz = PendulumRenderer(
        state_variable=pendulum.state,
        fps=60.0,
        window_size=(1200, 600),
        visible_history=1000,
    )

    # Reset every 1000 steps to see multiple swing-ups
    reset_node = ResetEachNSteps(
        node_name_to_reset=pendulum.external_name,
        n_steps=1000,
    )

    # Create and configure graph
    graph = Graph(
        [pendulum, controller, viz, reset_node],
        initialize_inner_time=True,
        states_to_log=[
            pendulum.state.full_name,
            controller.action.full_name,
        ],
        debug=True,
    )

    graph.resolve(graph.variables)

    # Run simulation
    n_steps = 10000
    for _ in range(n_steps):
        graph.step()
