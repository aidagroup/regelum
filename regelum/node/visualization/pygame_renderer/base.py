"""PyGame visualization node for system animation.

This module provides the base PyGame renderer and system-specific implementations.
The visualization system uses a multi-dashboard layout:
- Left: Custom system animation
- Center: State evolution plots
- Right (optional): Reward tracking

Key features:
- Real-time state plotting
- Configurable history length
- Automatic scaling and axes
- Optional reward tracking
- Custom system animations
- Multiple window support
"""

import os
from typing import Optional, List, Tuple, Dict, ClassVar
import ctypes

import pygame

from regelum import Node
from regelum import Variable
from regelum.node.core.types import NumericArray


class PyGameRenderer(Node):
    """Base PyGame renderer for system visualization.

    Provides a three-panel visualization layout:
    1. Animation Dashboard (left):
       - System-specific animation
       - Customizable through _render_animation_dashboard

    2. State Evolution (center):
       - Real-time plots of all state components
       - Automatic scaling and axis labels
       - Configurable history window

    3. Reward Evolution (right, optional):
       - Tracks reward signal over time
       - Useful for reinforcement learning

    The renderer automatically handles:
    - Window management and event processing
    - State history tracking and plotting
    - Time synchronization via FPS control
    - Resource cleanup
    - Multiple window support with automatic positioning

    Attributes:
        screen: PyGame display surface
        clock: FPS controller
        state_history: List of state trajectories
        reward_history: Optional reward trajectory
        visible_history: Number of points to show
        dashboard_width: Width of each panel
    """

    _initialized_pygame: ClassVar[bool] = False
    _active_windows: ClassVar[Dict[str, "PyGameRenderer"]] = {}
    _display_info: ClassVar[Optional[pygame.display.Info]] = None

    def __init__(
        self,
        state_variable: Variable,
        fps: float = 30.0,
        window_size: tuple[int, int] = (1600, 800),
        background_color: tuple[int, int, int] = (255, 255, 255),
        visible_history: int = 200,
        reward_variable: Optional[Variable] = None,
        delayed_init: bool = False,
        window_name: Optional[str] = None,
        window_position: Optional[Tuple[int, int]] = None,
    ):
        """Initialize PyGame renderer.

        Args:
            state_variable: Variable containing the state to visualize.
            fps: Frame rate for visualization.
            window_size: PyGame window size.
            background_color: Background color in RGB.
            visible_history: Number of points visible in the plot window.
            reward_variable: Optional variable containing the reward to track.
            delayed_init: Whether to delay the initialization of the renderer.
            window_name: Optional name for the window. If None, uses class name.
            window_position: Optional (x, y) position for the window. If None, auto-positions.
        """
        inputs = [state_variable.full_name]
        if reward_variable:
            inputs.append(reward_variable.full_name)

        super().__init__(inputs=inputs, name="pygame-renderer")
        self.initialized = False
        if not delayed_init:
            if not PyGameRenderer._initialized_pygame:
                pygame.init()
                pygame.font.init()
                PyGameRenderer._initialized_pygame = True
                PyGameRenderer._display_info = pygame.display.Info()

            self.font = pygame.font.SysFont("Arial", 14)

            # Set window name using Node's instance tracking
            self.window_name = (
                window_name
                or f"{self.__class__.__name__}_{len(self.__class__._instances[self.__class__.__name__])}"
            )

            # Set window position
            if window_position is None:
                # Auto-position windows in a grid based on active instances
                grid_size = 3  # 3x3 grid
                active_count = len(PyGameRenderer._active_windows)
                screen_width = PyGameRenderer._display_info.current_w
                screen_height = PyGameRenderer._display_info.current_h
                pos_x = (active_count % grid_size) * (screen_width // grid_size)
                pos_y = (active_count // grid_size) * (screen_height // grid_size)
                window_position = (pos_x, pos_y)

            # Create a new window with position
            self.screen = pygame.display.set_mode(
                window_size, pygame.RESIZABLE | pygame.SHOWN
            )
            pygame.display.set_caption(self.window_name)

            # Set window position after creation
            if hasattr(pygame.display, "get_wm_info"):
                try:
                    if os.name == "nt":  # Windows
                        hwnd = pygame.display.get_wm_info()["window"]
                        ctypes.windll.user32.SetWindowPos(
                            hwnd,
                            0,
                            window_position[0],
                            window_position[1],
                            0,
                            0,
                            0x0001,
                        )  # SWP_NOSIZE
                    else:  # Linux/X11
                        wminfo = pygame.display.get_wm_info()
                        if "window" in wminfo:
                            pos_str = f"{window_position[0]},{window_position[1]}"
                            os.environ["SDL_VIDEO_WINDOW_POS"] = pos_str
                            pygame.display.set_mode(
                                window_size, pygame.RESIZABLE | pygame.SHOWN
                            )
                except Exception:
                    pass  # Fallback if window positioning fails

            self.clock = pygame.time.Clock()
            self.fps = fps
            self.background_color = background_color
            self.initialized = True
            PyGameRenderer._active_windows[self.window_name] = self

        self.window_size = window_size
        self.state_variable = state_variable
        self.visible_history = visible_history
        self.state_history: List[List[float]] = []
        self.reward_history: List[float] = []
        self.current_step = 0
        self.has_reward = reward_variable is not None
        self.reward_variable = reward_variable

        if self.has_reward:
            self.dashboard_width = window_size[0] // 3
        else:
            self.dashboard_width = window_size[0] // 2
        self.dashboard_height = window_size[1]

    def _init_state_history(self, state_dim: int) -> None:
        """Initialize state history lists.

        Args:
            state_dim: Dimension of the state vector.
        """
        if not self.state_history:
            self.state_history = [[] for _ in range(state_dim)]

    def _render_animation_dashboard(self) -> None:
        """Base animation dashboard. Override in subclasses."""
        pass

    def _render_plots_dashboard(self, state: NumericArray) -> None:
        """Render right dashboard with state evolution plots.

        Args:
            state: Current state vector.
        """
        self._init_state_history(len(state))
        for i, value in enumerate(state):
            self.state_history[i].append(float(value))
        self.current_step += 1

        pygame.draw.rect(
            self.screen,
            (200, 200, 200),
            (self.dashboard_width, 0, self.dashboard_width, self.dashboard_height),
            2,
        )

        text = self.font.render("State Evolution", True, (0, 0, 0))
        text_rect = text.get_rect(center=(self.dashboard_width * 1.5, 30))
        self.screen.blit(text, text_rect)

        margin = 60
        plot_height = (self.dashboard_height - 2 * margin) // len(state)
        plot_margin = plot_height // 4

        for i, history in enumerate(self.state_history):
            if not history:
                continue

            plot_y = margin + i * plot_height
            plot_width = self.dashboard_width - 100
            plot_area_height = plot_height - plot_margin

            start_idx = max(0, len(history) - self.visible_history)
            visible_history = history[start_idx:]

            if visible_history:
                y_min = min(visible_history)
                y_max = max(visible_history)
                y_range = max(abs(y_max - y_min), 1e-6)
                y_min -= 0.1 * y_range
                y_max += 0.1 * y_range
            else:
                y_min, y_max = -1, 1

            pygame.draw.rect(
                self.screen,
                (240, 240, 240),
                (
                    self.dashboard_width + 50,
                    plot_y,
                    plot_width,
                    plot_area_height,
                ),
            )

            # Draw zero line first (if it's within the range)
            if y_min <= 0 <= y_max:
                zero_pixel = plot_y + plot_area_height * (
                    1 - (0 - y_min) / (y_max - y_min)
                )
                pygame.draw.line(
                    self.screen,
                    (150, 150, 150),  # Darker gray for better visibility
                    (self.dashboard_width + 50, zero_pixel),
                    (self.dashboard_width + 50 + plot_width, zero_pixel),
                    2,  # Thicker line for zero
                )

            # Draw grid lines
            n_grid_lines = 5
            for j in range(n_grid_lines):
                # Horizontal grid lines
                y_grid = y_min + (y_max - y_min) * j / (n_grid_lines - 1)
                y_pixel = plot_y + plot_area_height * (
                    1 - (y_grid - y_min) / (y_max - y_min)
                )
                pygame.draw.line(
                    self.screen,
                    (240, 240, 240),
                    (self.dashboard_width + 50, y_pixel),
                    (self.dashboard_width + 50 + plot_width, y_pixel),
                    1,
                )
                # Draw y-axis labels
                label = f"{y_grid:.2f}"
                text = self.font.render(label, True, (100, 100, 100))
                self.screen.blit(text, (self.dashboard_width + 20, y_pixel - 8))

            # Vertical grid lines (time)
            for j in range(n_grid_lines):
                x_grid = self.dashboard_width + 50 + j * plot_width / (n_grid_lines - 1)
                pygame.draw.line(
                    self.screen,
                    (240, 240, 240),
                    (x_grid, plot_y),
                    (x_grid, plot_y + plot_area_height),
                    1,
                )

            # Draw axis labels
            state_label = f"State {i}"
            text = self.font.render(state_label, True, (0, 0, 0))
            self.screen.blit(text, (self.dashboard_width + 50, plot_y - 20))

            if len(visible_history) > 1:
                points = []
                for t, value in enumerate(visible_history):
                    x = (
                        self.dashboard_width
                        + 50
                        + t * plot_width / (len(visible_history) - 1)
                    )
                    y = plot_y + plot_area_height * (
                        1 - (value - y_min) / (y_max - y_min)
                    )
                    points.append((int(x), int(y)))

                if len(points) > 1:
                    pygame.draw.lines(self.screen, (0, 0, 255), False, points, 2)

    def _render_reward_dashboard(self) -> None:
        """Render right dashboard with reward evolution plot."""
        if not self.has_reward:
            return

        x_start = 2 * self.dashboard_width
        pygame.draw.rect(
            self.screen,
            (200, 200, 200),
            (x_start, 0, self.dashboard_width, self.dashboard_height),
            2,
        )

        text = self.font.render("Reward Evolution", True, (0, 0, 0))
        text_rect = text.get_rect(center=(x_start + self.dashboard_width // 2, 30))
        self.screen.blit(text, text_rect)

        if not self.reward_history:
            return

        start_idx = max(0, len(self.reward_history) - self.visible_history)
        visible_history = self.reward_history[start_idx:]

        y_min = min(visible_history)
        y_max = max(visible_history)
        y_range = max(abs(y_max - y_min), 1e-6)
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range

        margin = 60
        plot_height = self.dashboard_height - 2 * margin
        plot_width = self.dashboard_width - 100

        # Draw plot background
        pygame.draw.rect(
            self.screen,
            (240, 240, 240),
            (x_start + 50, margin, plot_width, plot_height),
        )

        # Draw zero line if in range
        if y_min <= 0 <= y_max:
            zero_pixel = margin + plot_height * (1 - (0 - y_min) / (y_max - y_min))
            pygame.draw.line(
                self.screen,
                (150, 150, 150),
                (x_start + 50, zero_pixel),
                (x_start + 50 + plot_width, zero_pixel),
                2,
            )

        # Draw grid lines
        n_grid_lines = 5
        for j in range(n_grid_lines):
            # Horizontal grid lines
            y_grid = y_min + (y_max - y_min) * j / (n_grid_lines - 1)
            y_pixel = margin + plot_height * (1 - (y_grid - y_min) / (y_max - y_min))
            pygame.draw.line(
                self.screen,
                (240, 240, 240),
                (x_start + 50, y_pixel),
                (x_start + 50 + plot_width, y_pixel),
                1,
            )
            # Draw y-axis labels
            label = f"{y_grid:.2f}"
            text = self.font.render(label, True, (100, 100, 100))
            self.screen.blit(text, (x_start + 20, y_pixel - 8))

        # Vertical grid lines (time)
        for j in range(n_grid_lines):
            x_grid = x_start + 50 + j * plot_width / (n_grid_lines - 1)
            pygame.draw.line(
                self.screen,
                (240, 240, 240),
                (x_grid, margin),
                (x_grid, margin + plot_height),
                1,
            )

        points = []
        for t, value in enumerate(visible_history):
            x = (
                x_start
                + 50
                + (t * plot_width)
                // min(self.visible_history, len(self.reward_history))
            )
            normalized_y = (value - y_min) / (y_max - y_min)
            y = margin + plot_height - int(normalized_y * plot_height)
            y = max(margin, min(y, margin + plot_height))
            points.append((x, y))

        if len(points) > 1:
            pygame.draw.lines(self.screen, (0, 0, 255), False, points, 2)

    def _render_state(self, state: NumericArray) -> None:
        """Render state visualization with two dashboards.

        Args:
            state: System state vector.
        """
        self._render_animation_dashboard()
        self._render_plots_dashboard(state)

    def step(self) -> None:
        """Execute visualization step."""
        if not self.initialized:
            return

        # Handle events for this window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._close_window()
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._close_window()
                return
            if event.type == pygame.VIDEORESIZE:
                self.window_size = (event.w, event.h)
                self.screen = pygame.display.set_mode(
                    self.window_size, pygame.RESIZABLE | pygame.SHOWN
                )
                if self.has_reward:
                    self.dashboard_width = self.window_size[0] // 3
                else:
                    self.dashboard_width = self.window_size[0] // 2
                self.dashboard_height = self.window_size[1]

        self.screen.fill(self.background_color)
        state = self.resolved_inputs.find(self.inputs.inputs[0]).value

        if self.has_reward:
            reward = float(
                self.resolved_inputs.find(self.reward_variable.full_name).value
            )
            self.reward_history.append(reward)

        self._render_state(state)
        self._render_reward_dashboard()
        pygame.display.update()  # Update only this window
        self.clock.tick(self.fps)

    def _close_window(self) -> None:
        """Clean up resources and close the window."""
        if self.window_name in PyGameRenderer._active_windows:
            del PyGameRenderer._active_windows[self.window_name]
            self.initialized = False

        if not PyGameRenderer._active_windows:
            pygame.quit()  # Quit pygame if no windows remain
