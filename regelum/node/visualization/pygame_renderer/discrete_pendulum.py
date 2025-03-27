"""PyGame visualization node for discrete pendulum environment.

This module provides the DiscretePendulumRenderer class for visualizing a discrete pendulum.
The visualization includes:
- A pendulum animation with discretized states
- State plots for angle and velocity indices
- Optional reward tracking
"""

import pygame
import numpy as np

from regelum.node.visualization.pygame_renderer import PyGameRenderer


class DiscretePendulumRenderer(PyGameRenderer):
    """PyGame renderer for discrete pendulum visualization.

    Renders a pendulum with:
    - Fixed pivot point at center
    - Moving mass at end point
    - Discretized angle and velocity states
    - Grid showing discretization bins

    Layout:
    - Left: Pendulum animation with discretization grid
    - Center: State plots [angle_idx, velocity_idx]
    - Right (optional): Reward evolution
    """

    def __init__(
        self,
        *args: object,
        n_angles: int = 16,
        n_velocities: int = 16,
        max_velocity: float = 8.0,
        **kwargs: object,
    ):
        """Initialize discrete pendulum renderer.

        Args:
            n_angles: Number of angle discretization bins
            n_velocities: Number of velocity discretization bins
            max_velocity: Maximum velocity value
            *args: Additional positional arguments passed to PyGameRenderer
            **kwargs: Additional keyword arguments passed to PyGameRenderer
        """
        super().__init__(*args, **kwargs)
        self.n_angles = n_angles
        self.n_velocities = n_velocities
        self.max_velocity = max_velocity

        # Create discretization bins
        self.angle_bins = np.linspace(-np.pi, np.pi, n_angles + 1)
        self.velocity_bins = np.linspace(-max_velocity, max_velocity, n_velocities + 1)

    def _get_continuous_state(self) -> Tuple[float, float]:
        """Convert discrete indices to continuous state."""
        angle_idx = int(self.state_variable.value[0])
        velocity_idx = int(self.state_variable.value[1])

        # Use bin centers for visualization
        angle = (self.angle_bins[angle_idx] + self.angle_bins[angle_idx + 1]) / 2
        velocity = (
            self.velocity_bins[velocity_idx] + self.velocity_bins[velocity_idx + 1]
        ) / 2

        return angle, velocity

    def _render_animation_dashboard(self) -> None:
        """Render discrete pendulum animation in the left dashboard."""
        pygame.draw.rect(
            self.screen,
            (200, 200, 200),
            (0, 0, self.dashboard_width, self.dashboard_height),
            2,
        )

        center = (self.dashboard_width // 2, self.dashboard_height // 2)
        length = 200
        angle, velocity = self._get_continuous_state()

        # Draw angle discretization circles
        for angle_bin in self.angle_bins:
            radius = length * 0.8  # Slightly shorter than pendulum
            end_x = center[0] + radius * np.sin(angle_bin)
            end_y = center[1] - radius * np.cos(angle_bin)
            pygame.draw.circle(
                self.screen, (200, 200, 200), (int(end_x), int(end_y)), 3
            )

        # Draw pendulum
        end_pos = (
            center[0] + length * np.sin(angle),
            center[1] - length * np.cos(angle),
        )

        pygame.draw.circle(self.screen, (0, 0, 0), center, 10)  # Pivot
        pygame.draw.line(self.screen, (0, 0, 0), center, end_pos, 4)  # Rod
        pygame.draw.circle(
            self.screen, (255, 0, 0), (int(end_pos[0]), int(end_pos[1])), 20
        )  # Mass

        # Draw state information
        font = pygame.font.SysFont("Arial", 16)
        labels = [
            f"Angle Bin: {int(self.state_variable.value[0])} / {self.n_angles-1}",
            f"Velocity Bin: {int(self.state_variable.value[1])} / {self.n_velocities-1}",
            f"Angle: {angle:.2f}",
            f"Velocity: {velocity:.2f}",
        ]

        for i, label in enumerate(labels):
            text = font.render(label, True, (0, 0, 0))
            self.screen.blit(text, (20, 20 + i * 25))

        # Draw velocity indicator (arrow length proportional to velocity)
        if abs(velocity) > 0.1:  # Only draw if velocity is significant
            arrow_length = 50 * (velocity / self.max_velocity)
            arrow_x = end_pos[0] + arrow_length * np.cos(angle + np.pi / 2)
            arrow_y = end_pos[1] - arrow_length * np.sin(angle + np.pi / 2)
            pygame.draw.line(
                self.screen,
                (0, 0, 255),
                end_pos,
                (int(arrow_x), int(arrow_y)),
                3,
            )
