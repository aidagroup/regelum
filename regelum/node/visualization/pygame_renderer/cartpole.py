"""PyGame visualization node for cart-pole system.

This module provides the CartPoleRenderer class for visualizing a cart-pole system.
The visualization includes:
- A cart-pole animation
- State plots for angle and position
- Optional reward tracking
"""

import pygame
import numpy as np

from regelum.node.visualization.pygame_renderer import PyGameRenderer


class CartPoleRenderer(PyGameRenderer):
    """PyGame renderer for cart-pole system.

    Visualizes cart-pole with:
    - Moving cart on horizontal track
    - Pendulum pivoted on cart
    - State evolution plots

    Layout:
    - Left: Cart-pole animation
    - Center: State plots [θ, x, θ̇, ẋ]
    - Right (optional): Reward evolution
    """

    def _render_animation_dashboard(self) -> None:
        """Render cart-pole animation in the left dashboard."""
        pygame.draw.rect(
            self.screen,
            (200, 200, 200),
            (0, 0, self.dashboard_width, self.dashboard_height),
            2,
        )

        # Scale and center coordinates
        track_width = self.dashboard_width - 100
        scale = min(200, track_width / 4)
        center_x = self.dashboard_width // 2
        center_y = self.dashboard_height // 2

        # Cart dimensions
        cart_width = 100
        cart_height = 50
        pendulum_length = 200

        # Cart position (constrained to visible area)
        cart_x = np.clip(
            center_x + int(self.state_variable.value[1] * scale),
            cart_width // 2 + 50,
            self.dashboard_width - cart_width // 2 - 50,
        )
        cart_y = center_y

        # Draw track
        pygame.draw.line(
            self.screen,
            (100, 100, 100),
            (50, center_y + cart_height // 2),
            (self.dashboard_width - 50, center_y + cart_height // 2),
            4,
        )

        for x in range(-2, 3):  # -2m to +2m
            marker_x = center_x + x * scale
            if 50 <= marker_x <= self.dashboard_width - 50:
                pygame.draw.line(
                    self.screen,
                    (150, 150, 150),
                    (marker_x, center_y + cart_height // 2 - 10),
                    (marker_x, center_y + cart_height // 2 + 10),
                    2,
                )
                if x != 0:
                    label = f"{x}m"
                    text = self.font.render(label, True, (100, 100, 100))
                    text_rect = text.get_rect(center=(marker_x, center_y + cart_height))
                    self.screen.blit(text, text_rect)

        pygame.draw.rect(
            self.screen,
            (0, 0, 0),
            (
                cart_x - cart_width // 2,
                cart_y - cart_height // 2,
                cart_width,
                cart_height,
            ),
        )

        pendulum_end = (
            cart_x + pendulum_length * np.sin(self.state_variable.value[0]),
            cart_y - pendulum_length * np.cos(self.state_variable.value[0]),
        )
        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            (cart_x, cart_y),
            pendulum_end,
            6,
        )
        pygame.draw.circle(
            self.screen,
            (255, 0, 0),
            (int(pendulum_end[0]), int(pendulum_end[1])),
            15,
        )
