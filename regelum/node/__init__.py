"""Node system for building computational graphs.

This package provides a flexible framework for building and executing computational
graphs with a focus on control systems and simulation. It supports:

Core Components:
- Nodes: Self-contained computational units with state
- Variables: Type-safe data containers with reset capabilities
- Graphs: Containers that manage node execution and dependencies
- Parallel Execution: Task-based parallelism using Dask

Key Features:
1. Dependency Management:
   - Automatic resolution of node dependencies
   - Full variable name qualification (node_name.variable_name)
   - Circular dependency detection

2. Execution Control:
   - Mixed continuous/discrete time dynamics
   - Coordinated reset behavior
   - Parallel execution with dependency preservation

3. State Management:
   - Automatic variable tracking
   - Configurable reset behavior
   - Monte Carlo simulation support

Example:
    ```python
    from regelum.node import Node, Graph

    # Define a node
    class Controller(Node):
        def __init__(self):
            super().__init__(
                inputs=["plant_1.state"],
                step_size=0.01
            )
            self.action = self.define_variable("action", value=0.0)

        def step(self):
            # Implement control logic
            pass

    # Create and run a graph
    plant = Plant()
    controller = Controller()
    graph = Graph(
        [plant, controller],
        initialize_inner_time=True
    )
    graph.resolve(graph.variables)
    graph.step()
    ```

The package is designed for:
- Control system simulation
- Reinforcement learning environments
- Real-time data processing
- Distributed computation
"""
