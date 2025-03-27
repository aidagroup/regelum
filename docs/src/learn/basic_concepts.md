# Regelum Core Concepts

This tutorial covers the fundamental concepts of the Regelum framework, explaining how its component-based architecture works and how to use it effectively for simulation and control systems.

## Table of Contents
1. [Introduction to Regelum](#introduction-to-regelum)
2. [Node: The Basic Building Block](#node-the-basic-building-block)
3. [Variable: Managing State](#variable-managing-state)
4. [Inputs & ResolvedInputs: Handling Dependencies](#inputs--resolvedinputs-handling-dependencies)
5. [Graph: Orchestrating Execution](#graph-orchestrating-execution)
6. [ParallelGraph: Distributed Computation](#parallelgraph-distributed-computation)
7. [Putting It All Together: A Simple Example](#putting-it-all-together-a-simple-example)

## Introduction to Regelum

Regelum is a modular framework designed for building and simulating dynamic systems, with a particular focus on control systems. The framework is built around the concept of interconnected computational nodes that can be combined into complex graphs. This approach allows for:

- Modular, reusable components
- Clean separation of concerns
- Flexible composition of systems
- Automatic dependency resolution
- Simple time synchronization
- Parallel execution capabilities

## Node: The Basic Building Block

At the core of Regelum is the `Node` class, which represents a self-contained computational unit. Nodes encapsulate both state and behavior, making them the fundamental building blocks for constructing simulations.

### Key Node Concepts

- **State Management**: Each node manages its own internal state through `Variable` instances
- **Computational Logic**: Nodes implement their behavior in the `step()` method
- **Input/Output**: Nodes declare dependencies on other nodes' variables
- **Time Synchronization**: Nodes can operate in continuous or discrete time domains
- **Reset Behavior**: Nodes support state resetting for iterative simulations

### Creating a Custom Node

To create a custom node, subclass the `Node` base class and implement the required methods:

```python
from regelum import Node
import numpy as np

class MySystem(Node):
    def __init__(self, control_signal_name: str):
        # Initialize the node with dependencies and configurations
        super().__init__(
            inputs=[control_signal_name],  # Dependencies
            step_size=0.01,                # Time step
            is_continuous=True,            # Continuous-time dynamics
            name="my_system"               # Custom name
        )
        
        # Define the system state
        self.state = self.define_variable(
            "state", 
            value=np.array([0.0, 0.0]),
            metadata={"shape": (2,)}
        )
    
    def step(self) -> None:
        """Execute one computational step."""
        if self.resolved_inputs is None:
            return
            
        # Get the control input
        control = self.resolved_inputs.find(self.inputs.inputs[0]).value
        
        # Update the system state (simplified example)
        self.state.value += np.array([self.state.value[1], control[0]]) * self.step_size
        
    def state_transition_map(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Define continuous dynamics for the system (required for continuous nodes)"""
        return np.array([state[1], action[0]])
```

## Variable: Managing State

Variables are containers for data within nodes. They store values, track metadata, support reset operations, and enable symbolic computation through CasADi integration.

### Key Variable Concepts

- **Value Storage**: Variables store values (numpy arrays, torch tensors, etc.)
- **Shape Management**: Variables track their shape for consistency
- **Reset Behavior**: Variables can be reset to their initial values
- **Symbolic Computation**: Variables support symbolic representation for advanced control
- **Full Names**: Each variable has a unique full name (`node_name.variable_name`)

### Working with Variables

```python
# Creating a variable
self.position = self.define_variable(
    "position",                    # Local name
    value=np.array([0.0, 0.0]),    # Initial value
    metadata={
        "shape": (2,),             # Shape metadata
        "reset_modifier": lambda x: np.random.randn(2) * 0.1  # Custom reset behavior
    }
)

# Accessing the value
current_pos = self.position.value

# Setting the value
self.position.value = np.array([1.0, 2.0])

# Resetting the variable
self.position.reset()

# Getting the full name
full_name = self.position.full_name  # e.g., "robot_1.position"
```

## Inputs & ResolvedInputs: Handling Dependencies

Nodes interact with each other through inputs, which are references to variables owned by other nodes. The input system consists of two main classes:

1. **Inputs**: Unresolved dependencies specified as strings
2. **ResolvedInputs**: Actual variable references after resolution

### Key Input Concepts

- **Dependency Declaration**: Nodes declare their dependencies through inputs
- **Variable Resolution**: The framework resolves input dependencies to actual variables
- **Fuzzy Matching**: Input resolution supports fuzzy matching for flexibility
- **Dependency Validation**: The framework detects and reports circular dependencies

### Working with Inputs and ResolvedInputs

```python
# Declaring dependencies in __init__
super().__init__(inputs=["sensor_1.measurement", "controller_1.action"])

# Accessing input values in step()
if self.resolved_inputs is not None:
    measurement = self.resolved_inputs.find("sensor_1.measurement").value
    action = self.resolved_inputs.find("controller_1.action").value
    
    # Or using variable name only
    measurement = self.resolved_inputs.find("measurement").value
```

## Graph: Orchestrating Execution

The `Graph` class manages collections of nodes, coordinating their execution and handling dependencies. Graphs are special types of nodes themselves, enabling hierarchical composition.

### Key Graph Concepts

- **Node Management**: Graphs contain and manage collections of nodes
- **Dependency Resolution**: Graphs resolve dependencies between nodes
- **Execution Order**: Graphs determine the correct execution order based on dependencies
- **Time Synchronization**: Graphs coordinate time steps across nodes
- **Hierarchical Composition**: Graphs can contain other graphs, enabling hierarchy
- **Subgraph Detection**: Graphs can detect independent subgraphs for parallelization

### Creating and Using Graphs

```python
from regelum import Graph

# Create nodes
system = MySystem(control_signal_name="controller_1.action")
controller = MyController(system_state_name="my_system_1.state")
reset_node = ResetEachNSteps(node_name_to_reset="my_system_1", n_steps=1000)

# Create a graph
graph = Graph(
    [system, controller, reset_node],  # Nodes to manage
    initialize_inner_time=True,         # Add time management
    states_to_log=["my_system_1.state", "controller_1.action"], # Variables to log
    debug=True                          # Enable debug logging
)

# Resolve dependencies
graph.resolve(graph.variables)

# Run simulation
for _ in range(10000):
    graph.step()
```

### Graph Execution Order

Graphs execute nodes in a specific order based on their dependencies:

1. Root nodes (those that don't depend on others) execute first
2. Nodes that depend only on root nodes execute next
3. This continues until all nodes have executed

The graph handles automatic sorting to ensure nodes execute in the correct order.

## ParallelGraph: Distributed Computation

For computationally intensive simulations, Regelum provides the `ParallelGraph` class for distributed execution across multiple processes.

### Key ParallelGraph Concepts

- **Task-Based Parallelism**: Each node's `step()` becomes a separate Dask task
- **Dependency Preservation**: Task dependencies match node dependencies
- **Worker Management**: Tasks are distributed across worker processes
- **State Synchronization**: Node states are synchronized between tasks
- **Resource Management**: Workers, network connections, and resources are managed

### Using ParallelGraph

```python
from regelum import ParallelGraph

# Create the same nodes as before
# ...

# Create a parallel graph instead of a regular graph
graph = ParallelGraph(
    [system, controller, reset_node],
    n_workers=4,                   # Number of worker processes
    threads_per_worker=1,          # Threads per worker
    debug=True                     # Enable dashboard
)

# Resolve and run as normal
graph.resolve(graph.variables)
for _ in range(10000):
    graph.step()

# Clean up resources when done
graph.close()
```

## Putting It All Together: A Simple Example

Let's build a complete example of a pendulum control system:

```python
import numpy as np
from regelum import Node, Graph
from regelum.node.classic_control.envs.continuous import Pendulum
from regelum.node.classic_control.controllers.pid import PIDControllerBase
from regelum.node.reset import ResetEachNSteps

# Create the pendulum system
pendulum = Pendulum(control_signal_name="pid_controller_1.control_signal")

# Create a PD controller (PID with ki=0)
pd_controller = PIDControllerBase(
    pendulum.state,          # State to control
    idx_controlled_state=0,  # Control the angle (index 0)
    kp=20,                   # Proportional gain
    ki=0,                    # No integral action
    kd=20,                   # Derivative gain
    step_size=0.01           # Time step
)

# Create a reset node to periodically reset the pendulum
reset_node = ResetEachNSteps(
    node_name_to_reset=pendulum.external_name,
    n_steps=100
)

# Create a custom node to store and visualize the results
class ResultLogger(Node):
    def __init__(self, n_steps: int = 1000):
        super().__init__(
            inputs=["pendulum_1.state", "pid_controller_1.control_signal"],
            is_root=True
        )
        self.n_steps = n_steps
        self.states = []
        self.controls = []
        
    def step(self) -> None:
        if self.resolved_inputs is None:
            return
            
        state = self.resolved_inputs.find("pendulum_1.state").value
        control = self.resolved_inputs.find("pid_controller_1.control_signal").value
        
        self.states.append(state.copy())
        self.controls.append(control.copy())
        
        # Print progress
        if len(self.states) % 100 == 0:
            print(f"Step {len(self.states)}/{self.n_steps}")

result_logger = ResultLogger(n_steps=1000)

# Create the graph and connect everything
graph = Graph(
    [pendulum, pd_controller, reset_node, result_logger],
    initialize_inner_time=True,
    states_to_log=["pendulum_1.state", "pid_controller_1.control_signal"]
)

# Resolve dependencies
graph.resolve(graph.variables)

# Run the simulation
n_steps = 1000
for _ in range(n_steps):
    graph.step()

# Access the results
states = np.array(result_logger.states)
controls = np.array(result_logger.controls)

# Now you can plot or analyze the results
```

## Common Patterns and Best Practices

Here are some best practices for working with Regelum:

1. **Modularity**: Create specialized nodes for specific functions (systems, controllers, observers, visualizers)
2. **Proper Initialization**: Always initialize nodes with the correct inputs, step size, and continuity flags
3. **Error Handling**: Check if `resolved_inputs` is `None` in the `step()` method
4. **Node Naming**: Use descriptive node names to make debugging easier
5. **Variable Resetting**: Use reset modifiers to customize reset behavior
6. **Dependency Management**: Keep dependency chains simple and avoid circular dependencies
7. **Time Synchronization**: Use the smallest step size for critical components
8. **Graph Visualization**: Use `print(graph)` to visualize the node structure

## Conclusion

Regelum's component-based architecture provides a flexible and powerful framework for building complex simulation and control systems. By understanding the core concepts of Nodes, Variables, Inputs, and Graphs, you can create modular, reusable components that can be combined in different ways.

This tutorial covered the fundamental concepts, but Regelum offers many more advanced features like continuous-discrete integration, symbolic computation with CasADi, and parallel execution with Dask. Explore the other tutorials and examples to discover the full power of the framework. 