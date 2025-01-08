"""MPC controller implementation using SciPy optimizer."""

import numpy as np
from typing import Callable, Tuple
from scipy.optimize import minimize
from regelum.node.base import Node
from regelum.node.core.variable import Variable


class MPCContinuous(Node):
    """Model Predictive Controller using SciPy's optimization.

    This variant uses scipy.optimize.minimize for solving the optimal control problem.
    Key features:
    - RK4 integration for prediction
    - Sequential optimization formulation
    - Support for control bounds
    - Configurable prediction horizon
    """

    def __init__(
        self,
        controlled_system: Node,
        controlled_state: Variable,
        control_dimension: int,
        objective_function: Callable[[np.ndarray], float],
        control_bounds: Tuple[np.ndarray, np.ndarray],
        prediction_horizon: int = 20,
        step_size: float = 0.01,
    ):
        """Initialize MPC controller.

        Args:
            controlled_system: System to control
            controlled_state: State variable to track
            control_dimension: Number of control inputs
            objective_function: Cost function to minimize
            control_bounds: (lower_bounds, upper_bounds) for controls
            prediction_horizon: Number of steps to predict
            step_size: Integration time step
        """
        super().__init__(
            inputs=[controlled_state.full_name],
            name="mpc",
        )

        self.controlled_system = controlled_system
        self.controlled_state = controlled_state
        self.control_dimension = control_dimension
        self.objective_function = objective_function
        self.control_bounds = control_bounds
        self.prediction_horizon = prediction_horizon
        self.step_size = step_size

        # Initialize action variable
        self.action = self.define_variable(
            "mpc_action",
            value=np.zeros(control_dimension),
            metadata={"shape": (control_dimension,)},
        )

        # Create optimization problem
        self.optimization_problem = self._create_optimization_problem()
        self.u_prev = np.zeros(control_dimension * prediction_horizon)

    def _rk4_step(
        self, state: np.ndarray, control: np.ndarray, dt: float
    ) -> np.ndarray:
        """Perform one RK4 integration step.

        Args:
            state: Current state
            control: Control input
            dt: Time step

        Returns:
            Next state
        """
        k1 = self.controlled_system.state_transition_map(state, control)
        k2 = self.controlled_system.state_transition_map(state + dt / 2 * k1, control)
        k3 = self.controlled_system.state_transition_map(state + dt / 2 * k2, control)
        k4 = self.controlled_system.state_transition_map(state + dt * k3, control)
        return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def _simulate_trajectory(
        self, initial_state: np.ndarray, controls: np.ndarray
    ) -> np.ndarray:
        """Simulate system trajectory for given control sequence.

        Args:
            initial_state: Starting state
            controls: Control sequence [u₀, u₁, ..., uₙ]

        Returns:
            State trajectory [x₀, x₁, ..., xₙ]
        """
        state_dim = len(initial_state)
        trajectory = np.zeros((self.prediction_horizon + 1, state_dim))
        trajectory[0] = initial_state

        for t in range(self.prediction_horizon):
            control = controls[
                t * self.control_dimension : (t + 1) * self.control_dimension
            ]
            trajectory[t + 1] = self._rk4_step(trajectory[t], control, self.step_size)

        return trajectory

    def _objective(self, controls: np.ndarray, initial_state: np.ndarray) -> float:
        """Compute total cost for a control sequence.

        Args:
            controls: Flattened control sequence
            initial_state: Starting state

        Returns:
            Total cost over prediction horizon
        """
        trajectory = self._simulate_trajectory(initial_state, controls)

        # Sum stage costs
        total_cost = sum(self.objective_function(state) for state in trajectory)

        # Add control regularization
        total_cost += 1e-4 * np.sum(controls**2)

        return float(total_cost)

    def _create_optimization_problem(self) -> Callable[[np.ndarray], np.ndarray]:
        """Create the optimization problem.

        Returns:
            Function that solves one MPC step
        """
        return self.solve_mpc

    def solve_mpc(self, current_state: np.ndarray) -> np.ndarray:
        # Prepare bounds for all control inputs
        bounds = []
        for _ in range(self.prediction_horizon):
            for i in range(self.control_dimension):
                bounds.append((self.control_bounds[0][i], self.control_bounds[1][i]))

        # Solve optimization problem
        result = minimize(
            self._objective,
            x0=self.u_prev,  # Warm start
            args=(current_state,),
            method="SLSQP",
            bounds=bounds,
            options={"maxiter": 100, "ftol": 1e-6},
        )

        # Update warm start
        self.u_prev = result.x

        # Return first control action
        return result.x[: self.control_dimension]

    def step(self) -> None:
        """Execute one MPC step."""
        if self.resolved_inputs is None:
            return

        controlled_state = self.resolved_inputs.find(self.controlled_state.full_name)
        if controlled_state is not None:
            optimal_control = self.optimization_problem(controlled_state.value)
            self.action.value = optimal_control
