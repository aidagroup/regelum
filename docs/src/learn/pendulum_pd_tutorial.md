# PD Controller for Pendulum: Your First Regelum Project

This tutorial walks you through building your first control system with Regelum: a Proportional-Derivative (PD) controller for a pendulum. We'll implement a simulation that stabilizes an inverted pendulum and visualize the results.

## Prerequisites

- Basic understanding of control theory concepts
- Familiarity with Python
- Regelum framework installed

## Overview

The system we'll build consists of:

1. A pendulum simulation (plant)
2. A PD controller that stabilizes the pendulum
3. A reset mechanism to periodically disturb the system
4. A visualization component to record and plot the results

## Step 1: Import Required Libraries

First, let's import the necessary libraries:

```python
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from regelum import Node, Graph
from regelum.node.classic_control.envs.continuous import Pendulum
from regelum.node.classic_control.controllers.pid import PIDControllerBase
from regelum.node.reset import ResetEachNSteps
```

## Step 2: Create the Visualization Node

We'll create a custom node that records the system state and control signals, then plots them at the end of the simulation:

```python
class PlotDumper(Node):
    def __init__(self, save_dir: str = "plots", n_steps: int = 1000, **kwargs):
        inputs = [
            "pendulum_1.state",
            "pid_controller_1.control_signal",
        ]
        super().__init__(inputs=inputs, is_root=True, **kwargs)
        self.n_steps = n_steps
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.states = []
        self.controls = []

    def step(self) -> None:
        self.states.append(self.resolved_inputs.find("pendulum_1.state").value.copy())
        self.controls.append(
            self.resolved_inputs.find("pid_controller_1.control_signal").value
        )

        if len(self.states) == self.n_steps:
            self._dump_plot()

    def _dump_plot(self) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        states = np.array(self.states)
        controls = np.array(self.controls)
        times = np.arange(len(states)) * self.step_size

        ax1.plot(times, states[:, 0], label="angle")
        ax1.plot(times, states[:, 1], label="angular velocity")
        ax1.set_ylabel("State")
        ax1.legend()

        ax2.plot(times, controls, label="control")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Control")
        ax2.legend()

        plt.savefig(self.save_dir / "simulation.png")
        plt.close()
```

This node:
- Takes the pendulum state and controller signal as inputs
- Records these values at each step
- Creates a visualization once the simulation is complete

## Step 3: Instantiate the System Components

Now, let's create all the nodes required for our simulation:

```python
n_steps = 1000

# Create the pendulum node
pendulum = Pendulum(control_signal_name="pid_controller_1.control_signal")

# Create the PD controller
pd_controller = PIDControllerBase(
    pendulum.state, idx_controlled_state=0, kp=20, ki=0, kd=20, step_size=0.01
)

# Create a reset mechanism to periodically disturb the pendulum
reset_pendulum = ResetEachNSteps(node_name_to_reset=pendulum.external_name, n_steps=100)

# Create our visualization node
plot_dumper = PlotDumper(n_steps=n_steps)
```

Note how we:
1. Create a pendulum node, specifying its input from the controller
2. Create a PD controller (PID with no integral term), connecting it to the pendulum's state
3. Set up a reset mechanism that disturbs the pendulum every 100 steps
4. Create the visualization node to record the simulation

## Step 4: Create and Configure the Graph

The Graph is the execution container that manages all nodes:

```python
graph = Graph(
    [pendulum, pd_controller, reset_pendulum, plot_dumper],
    states_to_log=[
        "pendulum_1.state",
        "pid_controller_1.control_signal",
    ],
    initialize_inner_time=True,
    logger_cooldown=0.2,
)
```

Here we:
1. Add all our nodes to the graph
2. Specify which states to log during execution
3. Enable internal time tracking
4. Configure logging frequency

## Step 5: Resolve and Run the Simulation

Finally, we resolve all dependencies and run the simulation:

```python
# Resolve all dependencies between nodes
graph.resolve(graph.variables)

# Run the simulation for the specified number of steps
for _ in range(n_steps):
    graph.step()
```

This:
1. Resolves all node input dependencies
2. Runs the system for the specified number of steps
3. Automatically generates the visualization at the end

## Understanding the Results

After running this code, you'll find a `simulation.png` file in the `plots` directory. The plot will show:

- The pendulum's angle and angular velocity over time
- The control signal applied by the PD controller
- Periodic resets every 100 steps, visible as sudden changes in the pendulum state

![Simulation results showing pendulum stabilization with periodic resets](pendulum_pd_simulation.png)

## Key Concepts Demonstrated

This example demonstrates several key Regelum concepts:

1. **Node Composition**: Creating a system from independent, specialized nodes
2. **Dependency Management**: Automatically resolving dependencies between nodes
3. **Variable Access**: Accessing other nodes' variables through the input system
4. **Graph Execution**: Coordinating the execution of all nodes
5. **Reset Mechanisms**: Implementing periodic resets for testing robustness
6. **Visualization**: Recording and visualizing simulation results

## Further Exploration

Try modifying this example by:

- Changing the PD controller gains to observe different behaviors
- Adding an integral term to create a full PID controller
- Modifying the reset period or initial conditions
- Creating new visualization nodes with different plots
- Implementing a more complex controller strategy

## Conclusion

You've now built your first control system using Regelum! This framework makes it easy to construct complex control and simulation systems by combining specialized nodes into execution graphs.

By understanding this example, you have a foundation for creating more sophisticated systems using Regelum's powerful node-based architecture. 