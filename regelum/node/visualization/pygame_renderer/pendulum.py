"""PyGame visualization node for pendulum system.

This module provides the PendulumRenderer class for visualizing a pendulum system.
The visualization includes:
- A pendulum animation
- State plots for angle and angular velocity
- Optional reward tracking
"""

import pygame
import numpy as np

from regelum.node.visualization.pygame_renderer import PyGameRenderer
from regelum.node.core.types import NumericArray


class PendulumRenderer(PyGameRenderer):
    """PyGame renderer for pendulum system visualization.

    Renders a pendulum as a line with:
    - Fixed pivot point at center
    - Moving mass at end point
    - Angular position from state[0]

    The animation shows:
    - Black pivot point (fixed)
    - Black rod (pendulum arm)
    - Red mass (end effector)

    Layout:
    - Left: Pendulum animation
    - Center: State plots [angle, angular_velocity]
    - Right (optional): Reward evolution

    Example:
        ```python
        viz = PendulumRenderer(
            state_variable=pendulum.state,  # [theta, theta_dot]
            fps=60.0,
            window_size=(1200, 400)
        )
        ```
    """

    def _render_state(self, state: NumericArray) -> None:
        # Override animation dashboard with pendulum
        self._render_animation_dashboard()
        # Use parent's plot dashboard
        self._render_plots_dashboard(state)

    def _render_animation_dashboard(self) -> None:
        """Render pendulum animation in the left dashboard."""
        pygame.draw.rect(
            self.screen,
            (200, 200, 200),
            (0, 0, self.dashboard_width, self.dashboard_height),
            2,
        )

        center = (self.dashboard_width // 2, self.dashboard_height // 2)
        length = 200
        angle = self.state_variable.value[0]

        end_pos = (
            center[0] + length * np.sin(angle),
            center[1] - length * np.cos(angle),
        )

        pygame.draw.circle(self.screen, (0, 0, 0), center, 10)
        pygame.draw.line(self.screen, (0, 0, 0), center, end_pos, 4)
        pygame.draw.circle(
            self.screen, (255, 0, 0), (int(end_pos[0]), int(end_pos[1])), 20
        )
