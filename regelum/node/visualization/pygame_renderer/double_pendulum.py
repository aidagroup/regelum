"""PyGame visualization node for double pendulum system.

This module provides the DoublePendulumRenderer class for visualizing a double pendulum system.
The visualization includes:
- A double pendulum animation
- State plots for angles and angular velocities
- Optional reward tracking
"""

from typing import Optional, List, Tuple

import pygame
import numpy as np

from regelum.node.visualization.pygame_renderer import PyGameRenderer
from regelum import Variable


class DoublePendulumRenderer(PyGameRenderer):
    """PyGame renderer for double pendulum system visualization.

    Renders a double pendulum as two connected lines with:
    - Fixed pivot point at center
    - Two moving masses at end points
    - Angular positions from state[0] and state[1]

    The animation shows:
    - Black pivot point (fixed)
    - Black rods (pendulum arms)
    - Red mass (first pendulum)
    - Blue mass (second pendulum)
    - Optional motion trails

    Layout:
    - Left: Double pendulum animation
    - Center: State plots [θ₁, θ₂, ω₁, ω₂]
    - Right (optional): Reward evolution

    Example:
        ```python
        viz = DoublePendulumRenderer(
            state_variable=pendulum.state,  # [θ₁, θ₂, ω₁, ω₂]
            fps=60.0,
            window_size=(1200, 400)
        )
        ```
    """

    def __init__(
        self,
        state_variable: Variable,
        fps: float = 30.0,
        window_size: tuple[int, int] = (1600, 800),
        background_color: tuple[int, int, int] = (255, 255, 255),
        visible_history: int = 200,
        reward_variable: Optional[Variable] = None,
        trail_length: int = 50,
    ):
        """Initialize double pendulum renderer.

        Args:
            state_variable: Variable containing the state to visualize.
            fps: Frame rate for visualization.
            window_size: PyGame window size.
            background_color: Background color in RGB.
            visible_history: Number of points visible in the plot window.
            reward_variable: Optional variable containing the reward to track.
            trail_length: Number of previous positions to show in trail (0 to disable).
        """
        super().__init__(
            state_variable=state_variable,
            fps=fps,
            window_size=window_size,
            background_color=background_color,
            visible_history=visible_history,
            reward_variable=reward_variable,
        )
        self.trail_length = trail_length
        self.position_history: List[
            Tuple[Tuple[float, float], Tuple[float, float]]
        ] = []

    def _render_animation_dashboard(self) -> None:
        """Render double pendulum animation in the left dashboard."""
        pygame.draw.rect(
            self.screen,
            (200, 200, 200),
            (0, 0, self.dashboard_width, self.dashboard_height),
            2,
        )

        center = (self.dashboard_width // 2, self.dashboard_height // 2)
        scale = min(self.dashboard_width, self.dashboard_height) // 4

        theta1, theta2 = self.state_variable.value[0:2]

        x1 = center[0] + scale * np.sin(theta1)
        y1 = center[1] + scale * np.cos(theta1)
        x2 = x1 + scale * np.sin(theta2)
        y2 = y1 + scale * np.cos(theta2)

        if self.trail_length > 0:
            self.position_history.append(((x1, y1), (x2, y2)))
            if len(self.position_history) > self.trail_length:
                self.position_history.pop(0)

            for i, ((px1, py1), (px2, py2)) in enumerate(self.position_history[:-1]):
                alpha = int(255 * (i + 1) / len(self.position_history))
                trail_color = (100, 100, 100, alpha)
                trail_surface = pygame.Surface(
                    (self.dashboard_width, self.dashboard_height), pygame.SRCALPHA
                )
                pygame.draw.line(trail_surface, trail_color, (px1, py1), (px2, py2), 2)
                self.screen.blit(trail_surface, (0, 0))

        pygame.draw.line(self.screen, (0, 0, 0), center, (int(x1), int(y1)), 4)
        pygame.draw.line(
            self.screen, (0, 0, 0), (int(x1), int(y1)), (int(x2), int(y2)), 4
        )

        pygame.draw.circle(self.screen, (0, 0, 0), center, 8)  # Pivot
        pygame.draw.circle(
            self.screen, (255, 0, 0), (int(x1), int(y1)), 15
        )  # First mass
        pygame.draw.circle(
            self.screen, (0, 0, 255), (int(x2), int(y2)), 15
        )  # Second mass

        # Draw state labels
        font = pygame.font.SysFont("Arial", 16)
        labels = [
            f"θ₁: {theta1:.2f}",
            f"θ₂: {theta2:.2f}",
            f"ω₁: {self.state_variable.value[2]:.2f}",
            f"ω₂: {self.state_variable.value[3]:.2f}",
        ]
        for i, label in enumerate(labels):
            text = font.render(label, True, (0, 0, 0))
            self.screen.blit(text, (20, 20 + i * 25))
