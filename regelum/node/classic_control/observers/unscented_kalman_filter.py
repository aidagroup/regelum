"""Unscented Kalman Filter implementation for nonlinear state estimation."""

import numpy as np
from typing import Optional, Tuple
from regelum.node.base import Node
from regelum.node.core.types import NumericArray


class UnscentedKalmanFilter(Node):
    """Unscented Kalman Filter for nonlinear state estimation.

    The UKF uses sigma points to propagate the state distribution through
    nonlinear dynamics and measurement models. This implementation uses
    the scaled unscented transform with parameters alpha, beta, and kappa.

    Attributes:
        state_dim: Dimension of state vector
        meas_dim: Dimension of measurement vector
        alpha: Spread of sigma points around mean
        beta: Prior knowledge of state distribution (2 for Gaussian)
        kappa: Secondary scaling parameter
        Q: Process noise covariance
        R: Measurement noise covariance
    """

    def __init__(
        self,
        system_node: Node,
        measurement_node: Node,
        initial_state: Optional[NumericArray] = None,
        initial_covariance: Optional[NumericArray] = None,
        process_noise_cov: Optional[NumericArray] = None,
        measurement_noise_cov: Optional[NumericArray] = None,
        alpha: float = 0.1,
        beta: float = 2.0,
        kappa: float = 0.0,
        step_size: float = 0.01,
    ):
        """Initialize UKF.

        Args:
            system_node: Node containing system dynamics
            measurement_node: Node providing measurements
            initial_state: Initial state estimate
            initial_covariance: Initial state covariance
            process_noise_cov: Process noise covariance matrix
            measurement_noise_cov: Measurement noise covariance matrix
            alpha: Spread of sigma points (0 < alpha ≤ 1)
            beta: Prior knowledge parameter (2 optimal for Gaussian)
            kappa: Secondary scaling parameter (typically 0)
            step_size: Integration time step
        """
        # Get control input name from system node
        control_input_name = (
            system_node.inputs.inputs[0] if system_node.inputs.inputs else None
        )

        super().__init__(
            inputs=[measurement_node.observed_value.full_name]
            + ([control_input_name] if control_input_name else []),
            step_size=step_size,
            name="ukf",
        )

        self.system = system_node
        self.measurement = measurement_node
        self.has_control = bool(control_input_name)

        # Get dimensions
        self.state_dim = len(system_node.state.value)
        self.meas_dim = len(measurement_node.observed_value.value)

        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.n_sigma = 2 * self.state_dim + 1

        # Initialize state and covariance
        if initial_state is None:
            initial_state = np.zeros(self.state_dim)
        if initial_covariance is None:
            initial_covariance = np.eye(self.state_dim)
        if process_noise_cov is None:
            process_noise_cov = 0.1 * np.eye(self.state_dim)
        if measurement_noise_cov is None:
            measurement_noise_cov = 0.1 * np.eye(self.meas_dim)

        self.Q = process_noise_cov
        self.R = measurement_noise_cov

        # Create state estimate variable
        self.state_estimate = self.define_variable(
            "state_estimate",
            value=initial_state,
            shape=(self.state_dim,),
        )

        # Internal storage
        self.P = initial_covariance
        self._compute_weights()

    def _compute_weights(self) -> None:
        """Compute UKF weights for mean and covariance."""
        lambda_ = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim

        # Weights for mean
        self.Wm = np.zeros(self.n_sigma)
        self.Wm[0] = lambda_ / (self.state_dim + lambda_)
        self.Wm[1:] = 1 / (2 * (self.state_dim + lambda_))

        # Weights for covariance
        self.Wc = self.Wm.copy()
        self.Wc[0] += 1 - self.alpha**2 + self.beta

    def _generate_sigma_points(self, x: NumericArray, P: NumericArray) -> NumericArray:
        """Generate sigma points using scaled unscented transform.

        Args:
            x: State mean
            P: State covariance

        Returns:
            Array of sigma points
        """
        lambda_ = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim

        # Matrix square root using Cholesky decomposition
        try:
            L = np.linalg.cholesky((self.state_dim + lambda_) * P)
        except np.linalg.LinAlgError:
            # If Cholesky fails, add small diagonal term for numerical stability
            L = np.linalg.cholesky(
                (self.state_dim + lambda_) * P + 1e-8 * np.eye(self.state_dim)
            )

        # Generate sigma points
        sigma_points = np.zeros((self.n_sigma, self.state_dim))
        sigma_points[0] = x

        for i in range(self.state_dim):
            sigma_points[i + 1] = x + L[i]
            sigma_points[i + 1 + self.state_dim] = x - L[i]

        return sigma_points

    def _predict(
        self, control: Optional[NumericArray] = None
    ) -> Tuple[NumericArray, NumericArray]:
        """Predict step of UKF.

        Args:
            control: Optional control input for system dynamics

        Returns:
            Tuple of (predicted mean, predicted covariance)
        """
        # Generate sigma points
        sigma_points = self._generate_sigma_points(self.state_estimate.value, self.P)

        # Propagate sigma points through dynamics
        propagated_points = np.zeros_like(sigma_points)
        for i in range(self.n_sigma):
            if self.has_control:
                state_derivative = self.system.state_transition_map(
                    sigma_points[i], control
                )
            else:
                state_derivative = self.system.state_transition_map(sigma_points[i])

            propagated_points[i] = (
                sigma_points[i] + state_derivative.reshape(-1) * self.step_size
            )

        # Compute predicted mean
        x_pred = np.sum(self.Wm.reshape(-1, 1) * propagated_points, axis=0)

        # Compute predicted covariance
        P_pred = np.zeros((self.state_dim, self.state_dim))
        for i in range(self.n_sigma):
            diff = propagated_points[i] - x_pred
            P_pred += self.Wc[i] * np.outer(diff, diff)
        P_pred += self.Q

        return x_pred, P_pred

    def _update(
        self, x_pred: NumericArray, P_pred: NumericArray, measurement: NumericArray
    ) -> None:
        """Update step of UKF.

        Args:
            x_pred: Predicted state mean
            P_pred: Predicted state covariance
            measurement: Current measurement
        """
        # Generate sigma points from prediction
        sigma_points = self._generate_sigma_points(x_pred, P_pred)

        # Propagate sigma points through measurement model
        # Here we assume measurement model is just selecting observed states
        meas_sigma = np.zeros((self.n_sigma, self.meas_dim))
        observed_indices = self.measurement.observed_indices
        for i in range(self.n_sigma):
            meas_sigma[i] = sigma_points[i][observed_indices]

        # Predicted measurement mean
        y_pred = np.sum(self.Wm.reshape(-1, 1) * meas_sigma, axis=0)

        # Innovation covariance
        S = np.zeros((self.meas_dim, self.meas_dim))
        Pxy = np.zeros((self.state_dim, self.meas_dim))

        for i in range(self.n_sigma):
            diff_y = meas_sigma[i] - y_pred
            diff_x = sigma_points[i] - x_pred
            S += self.Wc[i] * np.outer(diff_y, diff_y)
            Pxy += self.Wc[i] * np.outer(diff_x, diff_y)
        S += self.R

        # Kalman gain
        K = Pxy @ np.linalg.inv(S)

        # Update state and covariance
        innovation = measurement - y_pred
        self.state_estimate.value = x_pred + K @ innovation
        self.P = P_pred - K @ S @ K.T

    def step(self) -> None:
        """Execute one step of the UKF."""
        if self.resolved_inputs is None:
            return

        # Get current measurement
        measurement = self.resolved_inputs.find(self.inputs.inputs[0]).value

        # Get control input if available
        control = None
        if self.has_control:
            control = self.resolved_inputs.find(self.inputs.inputs[1]).value

        # Predict
        x_pred, P_pred = self._predict(control)

        # Update
        self._update(x_pred, P_pred, measurement)
