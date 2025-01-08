"""Classic control systems and controllers.

This package implements traditional control system environments and controllers
as nodes. It includes:

Environments:
1. Pendulum Systems:
   - Single pendulum with torque control
     State: [θ, ω] (angle, angular velocity)
     Input: Torque [N⋅m]
     Parameters: Mass, length, gravity
   - Pendulum with motor dynamics
     State: [θ, ω, τ] (angle, angular velocity, motor torque)
     Input: Motor command [N⋅m/s]
     Parameters: Mass, length, gravity, motor mass, radius, time constant
   - Pendulum with friction
     State: [θ, ω] (angle, angular velocity)
     Input: Torque [N⋅m]
     Parameters: Mass, length, gravity, friction coefficient
   - Inverted pendulum on cart (CartPole)
     State: [θ, x, θ̇, ẋ] (angle, position, angular velocity, cart velocity)
     Input: Force on cart [N]
     Parameters: Cart mass, pendulum mass, length, gravity
   - Double pendulum dynamics
     State: [θ₁, θ₂, ω₁, ω₂] (angles and angular velocities)
     Input: [τ₁, τ₂] (torques on both joints)
     Parameters: Masses, lengths, gravity

2. Robot Systems:
   - Three-wheeled robot (kinematic and dynamic)
     Kinematic State: [x, y, θ] (position and orientation)
     Kinematic Input: [v, ω] (linear and angular velocity)
     Dynamic State: [x, y, θ, v, ω] (with velocities)
     Dynamic Input: [F, M] (force and moment)
     Parameters: Mass, inertia (dynamic model)
   - Kinematic point mass
     State: [x, y] (position)
     Input: [vx, vy] (velocities)

3. Linear Systems:
   - Mass-spring-damper
     State: [x, ẋ] (position, velocity)
     Input: External force [N]
     Parameters: Mass, spring constant, damping coefficient
   - DC motor
     State: [θ, ω, i] (angle, angular velocity, current)
     Input: Armature voltage [V]
     Parameters: Inertia, friction, motor constant, resistance, inductance

Controllers:
1. Classical Controllers:
   - PID with anti-windup
   - Linear Quadratic Regulator (LQR)
   - Energy-based swing-up control
   - Backstepping control

2. Model Predictive Control:
   - Continuous MPC with CasADi
   - Continuous MPC with SciPy

Features:
- Full state observation and control
- Configurable noise and disturbances
- Reset functionality for episodic learning
- CasADi integration for optimization
- Automatic differentiation support
- RK4 integration for continuous systems

Example:
    ```python
    from regelum.node.classic_control import Pendulum, MPCContinuous
    import numpy as np

    # Create pendulum system
    pendulum = Pendulum(
        control_signal_name="u",
        initial_state=np.array([np.pi, 0.0])
    )

    # Define objective function for MPC
    def objective(state):
        return 4 * state[0]**2 + state[1]**2

    # Create MPC controller
    mpc = MPCContinuous(
        controlled_system=pendulum,
        controlled_state=pendulum.state,
        control_dimension=1,
        objective_function=objective,
        control_bounds=(np.array([-2.0]), np.array([2.0])),
        prediction_horizon=20,
        step_size=0.01
    )

    # Connect and run
    graph = Graph(
        [pendulum, mpc],
        initialize_inner_time=True
    )
    graph.step()
    ```
"""
