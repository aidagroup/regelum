"""PyGame visualization node for grid world environment.

This module provides the GridWorldRenderer class for visualizing a grid world environment.
The visualization includes:
- A 2D grid animation with agent, obstacles, and goal
- State plots for position
- Optional reward tracking
"""

import pygame
from typing import List, Tuple

from regelum.node.visualization.pygame_renderer import PyGameRenderer


class GridWorldRenderer(PyGameRenderer):
    """PyGame renderer for grid world environment visualization.

    Renders a 2D grid world with:
    - Grid cells
    - Agent position
    - Obstacles
    - Goal position
    - State plots showing position

    Layout:
    - Left: Grid world animation
    - Center: State plots [row, col]
    - Right (optional): Reward evolution
    """

    def __init__(
        self,
        *args: object,
        grid_size: Tuple[int, int] = (5, 5),
        obstacles: List[Tuple[int, int]] = None,
        goal: Tuple[int, int] = (4, 4),
        **kwargs: object,
    ):
        """Initialize grid world renderer.

        Args:
            grid_size: Grid dimensions (rows, cols)
            obstacles: List of obstacle positions
            goal: Goal position
            *args: Additional positional arguments passed to PyGameRenderer
            **kwargs: Additional keyword arguments passed to PyGameRenderer
        """
        super().__init__(*args, **kwargs)
        self.grid_size = grid_size
        self.obstacles = obstacles or []
        self.goal = goal

    def _render_animation_dashboard(self) -> None:
        """Render grid world animation in the left dashboard."""
        pygame.draw.rect(
            self.screen,
            (200, 200, 200),
            (0, 0, self.dashboard_width, self.dashboard_height),
            2,
        )

        # Calculate cell dimensions and spacing
        margin = 50
        cell_width = min(
            80,
            min(
                (self.dashboard_width - 2 * margin) // self.grid_size[1],
                (self.dashboard_height - 2 * margin) // self.grid_size[0],
            ),
        )
        cell_height = cell_width
        total_width = cell_width * self.grid_size[1]
        total_height = cell_height * self.grid_size[0]
        start_x = (self.dashboard_width - total_width) // 2
        start_y = (self.dashboard_height - total_height) // 2

        # Draw grid cells
        for row in range(self.grid_size[0]):
            for col in range(self.grid_size[1]):
                x = start_x + col * cell_width
                y = start_y + row * cell_height
                rect = pygame.Rect(x, y, cell_width, cell_height)

                # Determine cell color
                if (row, col) == tuple(map(int, self.state_variable.value)):
                    color = (0, 0, 255)  # Agent (blue)
                elif (row, col) == self.goal:
                    color = (0, 255, 0)  # Goal (green)
                elif (row, col) in self.obstacles:
                    color = (128, 128, 128)  # Obstacle (gray)
                else:
                    color = (255, 255, 255)  # Empty cell (white)

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)

                # Draw coordinates
                text = self.font.render(f"({row},{col})", True, (0, 0, 0))
                text_rect = text.get_rect(
                    center=(x + cell_width // 2, y + cell_height // 2)
                )
                self.screen.blit(text, text_rect)

        # Draw legend
        legend_y = start_y - 40
        legend_items = [
            ("Agent", (0, 0, 255)),
            ("Goal", (0, 255, 0)),
            ("Obstacle", (128, 128, 128)),
        ]

        for i, (label, color) in enumerate(legend_items):
            x = start_x + i * 120
            pygame.draw.rect(self.screen, color, (x, legend_y, 20, 20))
            pygame.draw.rect(self.screen, (0, 0, 0), (x, legend_y, 20, 20), 1)
            text = self.font.render(label, True, (0, 0, 0))
            self.screen.blit(text, (x + 30, legend_y + 2))
