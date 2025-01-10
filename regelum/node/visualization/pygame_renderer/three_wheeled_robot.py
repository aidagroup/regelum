"""PyGame visualization node for three-wheeled robot system.

This module provides the ThreeWheeledRobotRenderer class for visualizing a three-wheeled robot system.
The visualization includes:
- A robot animation with orientation indicator
- State plots for x, y, and theta
- Optional reward tracking
"""

import pygame
import numpy as np

from regelum.node.visualization.pygame_renderer import PyGameRenderer


class ThreeWheeledRobotRenderer(PyGameRenderer):
    """PyGame renderer for three-wheeled robot system.

    Renders a mobile robot with:
    - Circular body with orientation indicator
    - Three wheels at 120° intervals
    - Coordinate grid for position reference
    - Robot state: [x, y, theta, ...]

    The visualization includes:
    - Gray circular robot body
    - Red direction indicator
    - Black wheels with proper orientation
    - Grid lines for position reference

    Layout:
    - Left: Robot animation with grid
    - Center: State plots [x, y, theta, ...]
    - Right (optional): Reward/cost evolution

    Example:
        ```python
        viz = ThreeWheeledRobotRenderer(
            state_variable=robot.state,
            window_size=(1600, 800),
            reward_variable=cost.value  # Track control cost
        )
        ```
    """

    def _render_animation_dashboard(self) -> None:
        """Render robot animation in the left dashboard.

        Args:
            state: Robot state [x, y, angle, ...].
        """
        pygame.draw.rect(
            self.screen,
            (200, 200, 200),
            (0, 0, self.dashboard_width, self.dashboard_height),
            2,
        )

        text = self.font.render("Robot Animation", True, (0, 0, 0))
        text_rect = text.get_rect(center=(self.dashboard_width // 2, 30))
        self.screen.blit(text, text_rect)

        scale = 100
        x = self.dashboard_width // 2 + int(self.state_variable.value[0] * scale)
        y = self.dashboard_height // 2 - int(self.state_variable.value[1] * scale)
        angle = self.state_variable.value[2]

        pygame.draw.line(
            self.screen,
            (200, 200, 200),
            (50, self.dashboard_height // 2),
            (self.dashboard_width - 50, self.dashboard_height // 2),
            2,
        )
        pygame.draw.line(
            self.screen,
            (200, 200, 200),
            (self.dashboard_width // 2, 50),
            (self.dashboard_width // 2, self.dashboard_height - 50),
            2,
        )

        robot_radius = 30
        wheel_width = 10
        wheel_length = 20

        pygame.draw.circle(self.screen, (100, 100, 100), (x, y), robot_radius)

        direction_x = x + int(robot_radius * np.cos(angle))
        direction_y = y - int(robot_radius * np.sin(angle))
        pygame.draw.line(
            self.screen, (255, 0, 0), (x, y), (direction_x, direction_y), 4
        )

        angles = [angle, angle + 2 * np.pi / 3, angle + 4 * np.pi / 3]
        for wheel_angle in angles:
            wheel_x = x + int(robot_radius * np.cos(wheel_angle))
            wheel_y = y - int(robot_radius * np.sin(wheel_angle))
            wheel_end_x = wheel_x + int(wheel_length * np.cos(wheel_angle + np.pi / 2))
            wheel_end_y = wheel_y - int(wheel_length * np.sin(wheel_angle + np.pi / 2))
            pygame.draw.line(
                self.screen,
                (50, 50, 50),
                (wheel_x, wheel_y),
                (wheel_end_x, wheel_end_y),
                wheel_width,
            )
