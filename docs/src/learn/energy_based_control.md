# Energy-Based Control Tutorial: Pendulum Swing-Up

This tutorial demonstrates how to implement an energy-based controller for pendulum swing-up using the regelum framework. Energy-based control is particularly useful for underactuated systems like pendulums, where we can utilize the system's natural dynamics to achieve control objectives efficiently.

## Overview

In this tutorial, you'll learn how to:

1. Set up a pendulum system
2. Implement an energy-based swing-up controller
3. Visualize the pendulum's behavior using PyGame
4. Configure a simulation graph to connect components
5. Run the simulation and analyze the results

Energy-based control for a pendulum is based on adding or removing energy from the system to reach the desired unstable equilibrium (upright position). This approach is more energy efficient than pure state feedback control for swing-up tasks.

## Prerequisites

Before starting, make sure you have regelum installed:

```bash
# Using pip
pip install -e .

# Using uv
uv pip install -e .
```

## The Pendulum System

Let's start by understanding our pendulum system. In regelum, physical systems are represented as nodes with well-defined states and dynamics.

```python
# Create pendulum system
pendulum = Pendulum(
    control_signal_name="energy_swing_up_1.action",
    initial_state=np.array([np.pi, 0.0]),  # Start from downward position
)
```

The pendulum state consists of:
- `x[0]`: Angle (θ) in radians, where 0 is upright and π is downward
- `x[1]`: Angular velocity (ω) in radians per second

We initialize the pendulum in the downward position (`np.pi`) with zero velocity.

## Creating a Reward Function

To evaluate the controller's performance, we define a reward function:

```python
class RewardPendulum(RewardTracker):
    """Reward function for pendulum stabilization."""

    @property
    def name(self) -> str:
        return "reward_pendulum"

    def objective_function(self, x: np.ndarray) -> float:
        angle_error = x[0]
        return float(4 * angle_error**2 + x[1] ** 2)
```

This reward function:
- Penalizes deviation from the upright position (θ = 0)
- Penalizes high angular velocities
- Uses a quadratic form to ensure smooth gradients

Let's create an instance of our reward tracker:

```python
# Create reward tracker
reward_tracker = RewardPendulum(state_variable=pendulum.state)
```

## Energy-Based Controller

Now for the key component—the energy-based controller. This controller works in two phases:
1. Swing-up: Adds or removes energy to reach the right energy level
2. Stabilization: Switches to PD control when near the upright position

```python
# Create energy-based controller with friction compensation
controller = EnergyBasedSwingUpController(
    controlled_state=pendulum.state,
    pendulum_params={
        "mass": pendulum.mass,
        "length": pendulum.length,
        "gravity": pendulum.gravity_acceleration,
        "friction": 0,  # Add friction coefficient
    },
    control_limits=(-10.0, 10.0),
    gain=2.0,
    pd_gains=(10.0, 1.0),  # Tune PD gains for stabilization
    switch_threshold=0.95,  # cos(theta) ≈ 18 degrees from vertical
)
```

The controller parameters:
- `control_limits`: Bounds the control signal to [-10, 10] Nm
- `gain`: Energy control gain (how aggressively to add/remove energy)
- `pd_gains`: Proportional and derivative gains for the stabilization phase
- `switch_threshold`: When to switch from energy control to PD control (0.95 ≈ 18° from vertical)

## Visualization

For real-time visualization, we use the PendulumRenderer class:

```python
# Create visualization
viz = PendulumRenderer(
    state_variable=pendulum.state,
    fps=60.0,
    window_size=(1200, 400),
    visible_history=1000,
    reward_variable=reward_tracker.reward,
)
```

This creates a PyGame window with:
- An animation of the pendulum motion (left panel)
- Plots of angle and angular velocity (middle panel)
- Plot of the reward over time (right panel)

## Reset Node

To observe multiple attempts at swing-up, we use a reset node:

```python
# Create reset node
reset_node = ResetEachNSteps(
    node_name_to_reset=pendulum.external_name,
    n_steps=1000,
)
```

This resets the pendulum to its initial state every 1000 steps.

## Setting Up the Graph

In regelum, components are connected through a Graph, which manages data flow and execution order:

```python
# Create and configure graph
graph = Graph(
    [pendulum, controller, viz, reset_node, reward_tracker],
    initialize_inner_time=True,
    states_to_log=[
        pendulum.state.full_name,
        controller.action.full_name,
        reward_tracker.reward.full_name,
        "step_counter_1.counter",
    ],
    logger_cooldown=0,
)
graph.resolve(graph.variables)
```

The graph:
- Connects all nodes (pendulum, controller, etc.)
- Sets up logging for state variables, control actions, and rewards
- Resolves references between nodes

## Running the Simulation

Finally, we run the simulation for 5000 steps:

```python
# Run simulation
n_steps = 5000
for _ in range(n_steps):
    graph.step()
```

Each call to `graph.step()` advances the simulation by one time step (0.01 seconds by default).

## Full Code Example

Here's the complete code for the energy-based pendulum swing-up:

```python
"""Example of pendulum swing-up using energy-based control."""

import numpy as np
from regelum.node.classic_control.envs.continuous import Pendulum
from regelum.node.classic_control.controllers.energy_based import (
    EnergyBasedSwingUpController,
)
from regelum.node.visualization.pygame_renderer import PendulumRenderer
from regelum.node.graph import Graph
from regelum.node.reset import ResetEachNSteps
from regelum.node.misc.reward import RewardTracker


class RewardPendulum(RewardTracker):
    """Reward function for pendulum stabilization."""

    @property
    def name(self) -> str:
        return "reward_pendulum"

    def objective_function(self, x: np.ndarray) -> float:
        angle_error = x[0]
        return float(4 * angle_error**2 + x[1] ** 2)


# Create pendulum system
pendulum = Pendulum(
    control_signal_name="energy_swing_up_1.action",
    initial_state=np.array([np.pi, 0.0]),  # Start from downward position
)

# Create reward tracker
reward_tracker = RewardPendulum(state_variable=pendulum.state)

# Create energy-based controller with friction compensation
controller = EnergyBasedSwingUpController(
    controlled_state=pendulum.state,
    pendulum_params={
        "mass": pendulum.mass,
        "length": pendulum.length,
        "gravity": pendulum.gravity_acceleration,
        "friction": 0,  # Add friction coefficient
    },
    control_limits=(-10.0, 10.0),
    gain=2.0,
    pd_gains=(10.0, 1.0),  # Tune PD gains for stabilization
    switch_threshold=0.95,  # cos(theta) ≈ 18 degrees from vertical
)

# Create visualization
viz = PendulumRenderer(
    state_variable=pendulum.state,
    fps=60.0,
    window_size=(1200, 400),
    visible_history=1000,
    reward_variable=reward_tracker.reward,
)

# Create reset node
reset_node = ResetEachNSteps(
    node_name_to_reset=pendulum.external_name,
    n_steps=1000,
)

# Create and configure graph
graph = Graph(
    [pendulum, controller, viz, reset_node, reward_tracker],
    initialize_inner_time=True,
    states_to_log=[
        pendulum.state.full_name,
        controller.action.full_name,
        reward_tracker.reward.full_name,
        "step_counter_1.counter",
    ],
    logger_cooldown=0,
)
graph.resolve(graph.variables)

# Run simulation
n_steps = 5000
for _ in range(n_steps):
    graph.step()
```

## How Energy-Based Control Works

Energy-based swing-up control works by manipulating the total energy of the pendulum system:

1. Calculate the current energy of the pendulum:
   ```
   E = 0.5 * m * L² * ω² + m * g * L * (1 - cos(θ))
   ```

2. Calculate the energy error:
   ```
   E_error = E - E_target
   ```
   where `E_target` is the energy at the unstable equilibrium (upright position).

3. Apply control based on energy error:
   ```
   u = -K * E_error * ω * cos(θ)
   ```
   This adds or removes energy based on the error.

4. When close to the upright position, switch to PD control:
   ```
   u = -Kp * θ - Kd * ω
   ```

## Experimenting with the Code

Try modifying the following parameters to observe different behaviors:

1. **Initial conditions**: Change the starting position of the pendulum
   ```python
   initial_state=np.array([np.pi/2, 0.0])  # Start from horizontal position
   ```

2. **Control gains**: Modify the energy or PD control gains
   ```python
   gain=1.0,  # Lower energy control gain
   pd_gains=(5.0, 0.5),  # Lower stabilization gains
   ```

3. **Reset interval**: Change how frequently the pendulum resets
   ```python
   n_steps=2000,  # Reset every 2000 steps
   ```

4. **Friction**: Add friction to make the system more realistic
   ```python
   "friction": 0.1,  # Add friction coefficient
   ```

## Conclusion

This tutorial demonstrated how to implement an energy-based controller for pendulum swing-up using regelum. The approach utilizes the natural dynamics of the system to efficiently bring the pendulum to its unstable equilibrium point, followed by stabilization using a PD controller.

Energy-based control is a powerful technique for underactuated systems and serves as a foundation for more complex control strategies. By understanding this example, you now have the tools to implement similar controllers for other mechanical systems and explore more advanced control techniques.

Try running the code and experimenting with different parameters to deepen your understanding of energy-based control and the regelum framework! 