"""Classic control systems and controllers.

This package implements traditional control system environments and controllers
as nodes. It includes:

Environments:
1. Pendulum Systems:
   - Single pendulum with torque control
   - Inverted pendulum on cart
   - Double pendulum dynamics

2. Robot Systems:
   - Three-wheeled robot with omnidirectional control
   - Kinematic point mass
   - Simple car model

3. Linear Systems:
   - Mass-spring-damper
   - DC motor

Controllers:
1. Classical Controllers:
   - PID with anti-windup
   - State feedback (LQR)
   - Feed-forward control

2. Model Predictive Control:
   - Linear MPC
   - Nonlinear MPC with CasADi
   - Explicit MPC implementations

3. Learning-Based Control:
   - Iterative Learning Control (ILC)
   - Model-based policy optimization
   - Adaptive control

Features:
- Full state observation and control
- Configurable noise and disturbances
- Reset functionality for episodic learning
- CasADi integration for optimization
- Automatic differentiation support

Example:
    ```python
    from regelum.node.classic_control import Pendulum, MPCController

    # Create environment and controller
    pendulum = Pendulum(
        state_init=np.array([np.pi, 0.0]),
        control_limits=(-2.0, 2.0)
    )

    controller = MPCController(
        horizon=20,
        Q=np.diag([1.0, 0.1]),
        R=np.array([[0.01]])
    )

    # Connect and run
    graph = Graph(
        [pendulum, controller],
        initialize_inner_time=True
    )
    graph.step()
    ```
"""
