"""Visualization nodes for system state and behavior.

This package provides visualization nodes that integrate with the computational graph
system to render system states and behaviors in real-time. Key features:

1. Real-time Visualization:
   - State evolution plots
   - Custom system animations
   - Reward tracking for RL systems
   - Configurable display layouts

2. PyGame-based Renderers:
   - Base renderer with customizable dashboards
   - System-specific renderers (Pendulum, Robot, etc.)
   - Automatic state history tracking
   - Multi-panel visualization support

3. Integration Features:
   - Seamless graph integration as nodes
   - Automatic state synchronization
   - Configurable update rates
   - Resource cleanup handling

Example:
    ```python
    from regelum.node.visualization import PendulumRenderer

    # Create visualization node
    viz = PendulumRenderer(
        state_variable=pendulum.state,
        fps=60.0,
        window_size=(800, 600),
        visible_history=200,
        reward_variable=reward_tracker.reward
    )

    # Add to computation graph
    graph = Graph([pendulum, controller, viz])
    graph.resolve(graph.variables)

    # Run simulation with visualization
    while True:
        graph.step()
    ```

Available Renderers:
- PyGameRenderer: Base class with state plots
- PendulumRenderer: Pendulum system visualization
- KinematicPointRenderer: 2D point system visualization
- ThreeWheeledRobotRenderer: Mobile robot visualization
"""
