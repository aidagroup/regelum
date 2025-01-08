"""PyGame visualization node for kinematic point system.

This module provides the KinematicPointRenderer class for visualizing a kinematic point system.
The visualization includes:
- A 2D point animation with grid background
- State plots for x and y coordinates
- Optional reward tracking
"""

import pygame

from regelum.node.visualization.pygame_renderer import PyGameRenderer
from regelum.node.core.types import NumericArray


class KinematicPointRenderer(PyGameRenderer):
    """PyGame renderer for 2D kinematic point system.

    Visualizes a point mass moving in 2D space with:
    - Coordinate grid background
    - Point position from state[0:2]
    - Origin at screen center
    - Scaled coordinates (100 pixels = 1 unit)

    Layout:
    - Left: 2D point animation with grid
    - Center: State plots [x, y, ...]
    - Right (optional): Reward evolution

    Example:
        ```python
        viz = KinematicPointRenderer(
            state_variable=point.state,  # [x, y, ...]
            visible_history=500  # Show longer trajectories
        )
        ```
    """

    def _render_state(self, state: NumericArray) -> None:
        self._render_animation_dashboard()
        self._render_plots_dashboard(state)

    def _render_animation_dashboard(self) -> None:
        """Render point animation in the left dashboard."""
        pygame.draw.rect(
            self.screen,
            (200, 200, 200),
            (0, 0, self.dashboard_width, self.dashboard_height),
            2,
        )

        scale = 100
        x = self.dashboard_width // 2 + int(self.state_variable[0] * scale)
        y = self.dashboard_height // 2 - int(self.state_variable[1] * scale)

        pygame.draw.line(
            self.screen,
            (200, 200, 200),
            (0, self.window_size[1] // 2),
            (self.window_size[0], self.window_size[1] // 2),
            2,
        )
        pygame.draw.line(
            self.screen,
            (200, 200, 200),
            (self.window_size[0] // 2, 0),
            (self.window_size[0] // 2, self.window_size[1]),
            2,
        )

        pygame.draw.circle(self.screen, (0, 0, 255), (x, y), 15)
