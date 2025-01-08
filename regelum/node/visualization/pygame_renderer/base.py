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
import numpy as np

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
        angle = self.state_variable[0]

        end_pos = (
            center[0] + length * np.sin(angle),
            center[1] - length * np.cos(angle),
        )

        pygame.draw.circle(self.screen, (0, 0, 0), center, 10)
        pygame.draw.line(self.screen, (0, 0, 0), center, end_pos, 4)
        pygame.draw.circle(
            self.screen, (255, 0, 0), (int(end_pos[0]), int(end_pos[1])), 20
        )


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
        x = self.dashboard_width // 2 + int(self.state_variable[0] * scale)
        y = self.dashboard_height // 2 - int(self.state_variable[1] * scale)
        angle = self.state_variable[2]

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
            center_x + int(self.state_variable[1] * scale),
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
            cart_x + pendulum_length * np.sin(self.state_variable[0]),
            cart_y - pendulum_length * np.cos(self.state_variable[0]),
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

        theta1, theta2 = self.state_variable[0:2]

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
            f"ω₁: {self.state_variable[2]:.2f}",
            f"ω₂: {self.state_variable[3]:.2f}",
        ]
        for i, label in enumerate(labels):
            text = font.render(label, True, (0, 0, 0))
            self.screen.blit(text, (20, 20 + i * 25))


class DCMotorRenderer(PyGameRenderer):
    """PyGame renderer for DC motor visualization.

    Provides a detailed visualization of a DC motor with:
    - Rotating motor shaft with position indicator
    - Current flow animation
    - Voltage indicator
    - Real-time electrical and mechanical state plots
    - Optional estimated state comparison

    Layout:
    - Left: Motor animation (mechanical + electrical)
    - Center: State plots [θ, ω, i]
    - Right (optional): Reward evolution
    """

    def __init__(
        self,
        state_variable: Variable,
        fps: float = 30.0,
        window_size: tuple[int, int] = (1600, 800),
        background_color: tuple[int, int, int] = (255, 255, 255),
        visible_history: int = 200,
        reward_variable: Optional[Variable] = None,
        show_current_flow: bool = True,
        estimated_state_variable: Optional[Variable] = None,
    ):
        """Initialize DC motor renderer.

        Args:
            state_variable: Variable containing the true state
            fps: Frame rate for visualization
            window_size: PyGame window size
            background_color: Background color in RGB
            visible_history: Number of points visible in the plot window
            reward_variable: Optional variable containing the reward to track
            show_current_flow: Whether to show current flow animation
            estimated_state_variable: Optional variable containing state estimates
        """
        inputs = [state_variable.full_name]
        if reward_variable:
            inputs.append(reward_variable.full_name)
        if estimated_state_variable:
            inputs.append(estimated_state_variable.full_name)

        super().__init__(
            state_variable=state_variable,
            fps=fps,
            window_size=window_size,
            background_color=background_color,
            visible_history=visible_history,
            reward_variable=reward_variable,
        )
        self.show_current_flow = show_current_flow
        self.current_flow_offset = 0
        self.estimated_state_variable = estimated_state_variable

    def _render_motor(
        self,
        state: NumericArray,
        center_x: int,
        center_y: int,
        motor_radius: int,
        color_scheme: tuple[tuple[int, int, int], ...],
    ) -> None:
        """Render a single motor visualization.

        Args:
            state: Motor state [θ, ω, i]
            center_x: X coordinate of motor center
            center_y: Y coordinate of motor center
            motor_radius: Radius of motor visualization
            color_scheme: Tuple of colors for (housing, rotor, indicator)
        """
        theta, _, _ = state[0], state[1], state[2]
        housing_color, rotor_color, indicator_color = color_scheme

        # Draw motor housing (stator)
        pygame.draw.circle(
            self.screen, housing_color, (center_x, center_y), motor_radius
        )
        pygame.draw.circle(
            self.screen, (200, 200, 200), (center_x, center_y), motor_radius - 4
        )

        indicator_length = motor_radius - 8
        end_x = center_x + indicator_length * np.cos(theta)
        end_y = center_y + indicator_length * np.sin(theta)
        pygame.draw.line(
            self.screen, rotor_color, (center_x, center_y), (end_x, end_y), 6
        )

        shaft_width = 20
        pygame.draw.rect(
            self.screen,
            indicator_color,
            (
                center_x - shaft_width // 2,
                center_y - motor_radius - 10,
                shaft_width,
                20,
            ),
        )
        pygame.draw.rect(
            self.screen,
            indicator_color,
            (
                center_x - shaft_width // 2,
                center_y + motor_radius - 10,
                shaft_width,
                20,
            ),
        )

    def _render_animation_dashboard(self) -> None:
        """Render DC motor animation in the left dashboard."""
        # Draw dashboard border
        pygame.draw.rect(
            self.screen,
            (200, 200, 200),
            (0, 0, self.dashboard_width, self.dashboard_height),
            2,
        )

        motor_radius = min(self.dashboard_width // 3, self.dashboard_height // 3)

        true_center_x = self.dashboard_width // 3
        true_center_y = self.dashboard_height // 2
        self._render_motor(
            self.state_variable.value,
            true_center_x,
            true_center_y,
            motor_radius,
            ((150, 150, 150), (100, 100, 100), (50, 50, 50)),
        )

        # Draw estimated state motor if available (right side)
        if self.estimated_state_variable and len(self.inputs.inputs) > 2:
            est_state = self.resolved_inputs.find(self.inputs.inputs[2]).value
            est_center_x = 2 * self.dashboard_width // 3
            est_center_y = self.dashboard_height // 2
            self._render_motor(
                est_state,
                est_center_x,
                est_center_y,
                motor_radius,
                ((100, 150, 100), (50, 100, 50), (0, 80, 0)),
            )

            # Draw labels
            font = pygame.font.SysFont("Arial", 16)
            true_label = font.render("True State", True, (0, 0, 0))
            est_label = font.render("Estimated State", True, (0, 0, 0))
            self.screen.blit(
                true_label, (true_center_x - 40, true_center_y - motor_radius - 30)
            )
            self.screen.blit(
                est_label, (est_center_x - 50, est_center_y - motor_radius - 30)
            )

        font = pygame.font.SysFont("Arial", 16)
        theta, omega, current = (
            self.state_variable.value[0],
            self.state_variable.value[1],
            self.state_variable.value[2],
        )
        labels = [
            f"θ: {theta:.2f} rad",
            f"ω: {omega:.2f} rad/s",
            f"i: {current:.2f} A",
        ]
        for i, label in enumerate(labels):
            text = font.render(label, True, (0, 0, 0))
            self.screen.blit(text, (20, 20 + i * 25))

        if self.estimated_state_variable and len(self.inputs.inputs) > 2:
            est_state = self.resolved_inputs.find(self.inputs.inputs[2]).value
            est_labels = [
                f"θ̂: {est_state[0]:.2f} rad",
                f"ω̂: {est_state[1]:.2f} rad/s",
                f"î: {est_state[2]:.2f} A",
            ]
            for i, label in enumerate(est_labels):
                text = font.render(label, True, (0, 100, 0))
                self.screen.blit(text, (self.dashboard_width - 150, 20 + i * 25))

    def _render_plots_dashboard(self, state: NumericArray) -> None:
        """Override to add estimated state plots."""
        super()._render_plots_dashboard(state)

        # Add estimated state plots if available
        if self.estimated_state_variable and len(self.inputs.inputs) > 2:
            est_state = self.resolved_inputs.find(self.inputs.inputs[2]).value
            for i, value in enumerate(est_state):
                y = value
                self.state_history[i].append(float(y))

                if len(self.state_history[i]) > 1:
                    points = []
                    start_idx = max(
                        0, len(self.state_history[i]) - self.visible_history
                    )
                    visible_history = self.state_history[i][start_idx:]

                    if visible_history:
                        y_min = min(visible_history)
                        y_max = max(visible_history)
                        y_range = max(abs(y_max - y_min), 1e-6)
                        y_min -= 0.1 * y_range
                        y_max += 0.1 * y_range

                        for t, value in enumerate(visible_history):
                            x = (
                                self.dashboard_width
                                + 50
                                + t
                                * (self.dashboard_width - 100)
                                / (len(visible_history) - 1)
                            )
                            plot_y = (
                                60
                                + i * ((self.dashboard_height - 2 * 60) // len(state))
                                + ((self.dashboard_height - 2 * 60) // len(state) - 20)
                                * (1 - (value - y_min) / (y_max - y_min))
                            )
                            points.append((int(x), int(plot_y)))

                        if len(points) > 1:
                            pygame.draw.lines(
                                self.screen, (0, 150, 0), False, points, 1
                            )
