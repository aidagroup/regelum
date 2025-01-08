"""MPC controller demonstration."""

from regelum.node.base import Node
from regelum.node.core.variable import Variable
from enum import Enum
import numpy as np
from typing import Callable, Tuple
import casadi as ca
from regelum import symbolic_mode


class MPCContinuous(Node):
    """Generic MPC controller implementation."""

    action_name = "mpc_action"

    class Engine(Enum):
        """MPC solver engine."""

        CASADI = "casadi"
        SCIPY = "scipy"

    def __init__(
        self,
        controlled_system: Node,
        controlled_state: Variable,
        control_dimension: int,
        objective_function: Callable[[np.ndarray], float],
        control_bounds: Tuple[np.ndarray, np.ndarray],
        step_size: float = 0.01,
        prediction_horizon: int = 3,
        name: str = "mpc",
    ):
        """Initialize MPC controller.

        Args:
            controlled_system: Controlled system.
            controlled_state: Controlled state.
            control_dimension: Control dimension.
            objective_function: Objective function.
            control_bounds: Control bounds.
            step_size: Step size.
            prediction_horizon: Prediction horizon.
            engine: MPC solver engine.
            name: Name of the MPC controller.
        """
        assert controlled_system.is_continuous
        super().__init__(
            is_root=False,
            step_size=step_size,
            is_continuous=False,
            inputs=[controlled_state.full_name],
            name=name,
        )
        self.controlled_system = controlled_system
        self.controlled_state = controlled_state
        self.control_dimension = control_dimension
        self.prediction_horizon = prediction_horizon
        self.objective_function = objective_function
        self.control_bounds = control_bounds
        self.optimization_problem: Callable[[np.ndarray], np.ndarray] = (
            self._create_optimization_problem()
        )
        self.action = self.define_variable(
            "mpc_action",
            shape=(self.control_dimension,),
            value=np.zeros(self.control_dimension),
        )
        self._assert_shapes()

    def _assert_shapes(self) -> None:
        """Assert that the controlled state has valid metadata."""
        if not hasattr(self.controlled_state, "metadata") or not isinstance(
            self.controlled_state.metadata, dict
        ):
            raise ValueError(
                "Controlled state must have metadata with shape information"
            )

        state_shape = self.controlled_state.metadata.get("shape")

        if not isinstance(state_shape, tuple) or len(state_shape) == 0:
            raise ValueError("Invalid state shape in metadata")

    def _create_optimization_problem(self) -> Callable[[np.ndarray], np.ndarray]:
        state_shape = self.controlled_state.metadata.get("shape")
        state_dim = int(state_shape[0])

        dt = self.step_size
        N = self.prediction_horizon

        opti = ca.Opti()
        X = opti.variable(state_dim, N + 1)
        U = opti.variable(self.control_dimension, N)

        cost = 0
        for k in range(N):
            cost += self.objective_function(X[:, k])

        cost += self.objective_function(X[:, N])

        opti.minimize(cost)

        x0 = opti.parameter(state_dim)
        opti.subject_to(X[:, 0] == x0)

        for k in range(N):
            # RK4 integration using the system's state transition map
            x_k = X[:, k]
            u_k = U[:, k]
            with symbolic_mode():
                k1 = self.controlled_system.state_transition_map(x_k, u_k)
                k2 = self.controlled_system.state_transition_map(x_k + dt / 2 * k1, u_k)
                k3 = self.controlled_system.state_transition_map(x_k + dt / 2 * k2, u_k)
                k4 = self.controlled_system.state_transition_map(x_k + dt * k3, u_k)

            x_next = x_k + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            opti.subject_to(X[:, k + 1] == x_next)

        u_min = self.control_bounds[0][:, None]
        u_max = self.control_bounds[1][:, None]
        opti.subject_to(ca.vec(U) >= ca.vec(ca.DM(np.repeat(u_min, N, axis=1))))
        opti.subject_to(ca.vec(U) <= ca.vec(ca.DM(np.repeat(u_max, N, axis=1))))

        opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", opts)

        self.opti = opti
        self.x0 = x0
        self.U = U

        return self.solve_mpc

    def solve_mpc(self, current_state: np.ndarray) -> np.ndarray:
        self.opti.set_value(self.x0, current_state)
        sol = self.opti.solve()
        return sol.value(self.U[:, 0])

    def step(self) -> None:
        if self.resolved_inputs is None:
            return

        controlled_state = self.resolved_inputs.find(self.controlled_state.full_name)
        if controlled_state is not None:
            optimal_control = self.optimization_problem(controlled_state.value)
            self.action.value = optimal_control
