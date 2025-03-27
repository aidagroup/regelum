"""PyGame visualization node for chain environment.

This module provides the ChainRenderer class for visualizing a chain environment.
The visualization includes:
- A 1D chain animation with current position
- State plot for position
- Optional reward tracking
"""

import pygame

from regelum.node.visualization.pygame_renderer import PyGameRenderer


class ChainRenderer(PyGameRenderer):
    """PyGame renderer for chain environment visualization.

    Renders a 1D chain with:
    - Horizontal chain of cells
    - Current position highlighted
    - Goal position marked
    - State plot showing position

    Layout:
    - Left: Chain animation
    - Center: State plot [position]
    - Right (optional): Reward evolution
    """

    def __init__(
        self, *args: object, length: int = 10, goal: int = 9, **kwargs: object
    ):
        """Initialize chain renderer.

        Args:
            length: Length of the chain
            goal: Goal position
            *args: Additional positional arguments passed to PyGameRenderer
            **kwargs: Additional keyword arguments passed to PyGameRenderer
        """
        super().__init__(*args, **kwargs)
        self.length = length
        self.goal = goal

    def _render_animation_dashboard(self) -> None:
        """Render chain animation in the left dashboard."""
        pygame.draw.rect(
            self.screen,
            (200, 200, 200),
            (0, 0, self.dashboard_width, self.dashboard_height),
            2,
        )

        # Calculate cell dimensions and spacing
        margin = 50
        cell_width = min(80, (self.dashboard_width - 2 * margin) // self.length)
        cell_height = cell_width
        total_width = cell_width * self.length
        start_x = (self.dashboard_width - total_width) // 2
        y = self.dashboard_height // 2

        # Draw cells
        for i in range(self.length):
            x = start_x + i * cell_width
            rect = pygame.Rect(x, y - cell_height // 2, cell_width, cell_height)

            # Determine cell color
            if i == int(self.state_variable.value[0]):
                color = (0, 0, 255)  # Current position (blue)
            elif i == self.goal:
                color = (0, 255, 0)  # Goal (green)
            else:
                color = (255, 255, 255)  # Empty cell (white)

            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)

            # Draw position number
            text = self.font.render(str(i), True, (0, 0, 0))
            text_rect = text.get_rect(center=(x + cell_width // 2, y + cell_height))
            self.screen.blit(text, text_rect)

        # Draw legend
        legend_y = y - cell_height * 2
        legend_items = [
            ("Current", (0, 0, 255)),
            ("Goal", (0, 255, 0)),
        ]

        for i, (label, color) in enumerate(legend_items):
            x = start_x + i * 120
            pygame.draw.rect(self.screen, color, (x, legend_y, 20, 20))
            pygame.draw.rect(self.screen, (0, 0, 0), (x, legend_y, 20, 20), 1)
            text = self.font.render(label, True, (0, 0, 0))
            self.screen.blit(text, (x + 30, legend_y + 2))
