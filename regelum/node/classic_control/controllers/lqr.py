"""Linear Quadratic Regulator (LQR) controller implementation."""

from typing import Optional, Tuple
import numpy as np
from scipy import linalg
from regelum.node.base import Node
from regelum import Variable


class LQRController(Node):
    """Linear Quadratic Regulator for continuous-time linear systems.

    Implements infinite-horizon LQR control for systems of the form:
    dx/dt = Ax + Bu

    The controller minimizes the quadratic cost:
    J = integral(x'Qx + u'Ru) dt

    Features:
    - Automatic gain computation from system matrices
    - Optional reference tracking
    - Continuous-time implementation
    - Anti-windup for control constraints

    Example:
        ```python
        # For a double integrator system
        A = np.array([[0, 1], [0, 0]])
        B = np.array([[0], [1]])
        Q = np.diag([1.0, 0.1])  # State cost
        R = np.array([[0.01]])    # Control cost

        lqr = LQRController(
            controlled_state=system.state,
            system_matrices=(A, B),
            cost_matrices=(Q, R),
            control_limits=(-1.0, 1.0)
        )
        ```
    """

    def __init__(
        self,
        controlled_state: Variable,
        system_matrices: Tuple[np.ndarray, np.ndarray],
        cost_matrices: Tuple[np.ndarray, np.ndarray],
        control_limits: Optional[Tuple[float, float]] = None,
        reference: Optional[Variable] = None,
        step_size: float = 0.01,
        name: str = "lqr",
    ):
        """Initialize LQR controller.

        Args:
            controlled_state: System state variable
            system_matrices: Tuple of (A, B) matrices
            cost_matrices: Tuple of (Q, R) cost matrices
            control_limits: Optional control bounds (u_min, u_max)
            reference: Optional reference state for tracking
            step_size: Controller update interval
            name: Controller name
        """
        inputs = [controlled_state.full_name]
        if reference is not None:
            inputs.append(reference.full_name)

        super().__init__(
            inputs=inputs,
            step_size=step_size,
            is_continuous=False,
            name=name,
        )

        # System matrices
        self.A, self.B = system_matrices
        self.Q, self.R = cost_matrices

        # Validate dimensions
        state_dim = controlled_state.metadata["shape"][0]
        control_dim = self.B.shape[1]
        assert self.A.shape == (state_dim, state_dim)
        assert self.B.shape == (state_dim, control_dim)
        assert self.Q.shape == (state_dim, state_dim)
        assert self.R.shape == (control_dim, control_dim)

        # Compute control gain
        self.K = self._solve_dare()

        # Initialize variables
        self.action = self.define_variable(
            "action",
            value=np.zeros(control_dim),
            metadata={"shape": (control_dim,)},
        )

        self.reference = reference
        if reference is None:
            self.reference = self.define_variable(
                "reference",
                value=np.zeros(state_dim),
                metadata={"shape": (state_dim,)},
            )

        self.control_limits = control_limits

    def _solve_dare(self) -> np.ndarray:
        """Solve the discrete algebraic Riccati equation.

        Returns:
            LQR gain matrix K
        """
        # Convert to discrete time system (ZOH approximation)
        dt = self.step_size
        Ad = linalg.expm(self.A * dt)
        Bd = np.linalg.solve(self.A, (Ad - np.eye(self.A.shape[0]))) @ self.B

        P = linalg.solve_discrete_are(Ad, Bd, self.Q * dt, self.R * dt)
        K = -np.linalg.solve(self.R * dt + Bd.T @ P @ Bd, Bd.T @ P @ Ad)

        return K

    def step(self) -> None:
        """Execute one control step."""
        if self.resolved_inputs is None:
            return

        # Get current state
        state_var = self.resolved_inputs.find(self.inputs.inputs[0])
        if state_var is None:
            return

        state = state_var.value
        reference = self.reference.value

        # Compute control action
        error = state - reference
        u = self.K @ error

        # Apply control limits if specified
        if self.control_limits is not None:
            u = np.clip(u, self.control_limits[0], self.control_limits[1])

        self.action.value = u
