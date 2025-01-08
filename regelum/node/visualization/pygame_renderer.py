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
"""

from typing import Optional, List
import pygame
import numpy as np
from regelum import Node
from regelum.node.core.variable import Variable
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

    Attributes:
        screen: PyGame display surface
        clock: FPS controller
        state_history: List of state trajectories
        reward_history: Optional reward trajectory
        visible_history: Number of points to show
        dashboard_width: Width of each panel
    """

    def __init__(
        self,
        state_variable: Variable,
        fps: float = 30.0,
        window_size: tuple[int, int] = (1600, 800),
        background_color: tuple[int, int, int] = (255, 255, 255),
        visible_history: int = 200,
        reward_variable: Optional[Variable] = None,
    ):
        """Initialize PyGame renderer.

        Args:
            state_variable: Variable containing the state to visualize.
            fps: Frame rate for visualization.
            window_size: PyGame window size.
            background_color: Background color in RGB.
            visible_history: Number of points visible in the plot window.
            reward_variable: Optional variable containing the reward to track.
        """
        inputs = [state_variable.full_name]
        if reward_variable:
            inputs.append(reward_variable.full_name)

        super().__init__(inputs=inputs, name="pygame-renderer")
        pygame.init()
        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 14)
        self.screen = pygame.display.set_mode(window_size)
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.background_color = background_color
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

    def _render_animation_dashboard(self, state: NumericArray) -> None:
        """Render left dashboard with custom animation if implemented.

        Args:
            state: Current state vector.
        """
        # Draw dashboard border
        pygame.draw.rect(
            self.screen,
            (200, 200, 200),
            (0, 0, self.dashboard_width, self.dashboard_height),
            2,
        )
        # Default implementation is empty
        text = self.font.render("Animation Dashboard", True, (0, 0, 0))
        text_rect = text.get_rect(center=(self.dashboard_width // 2, 30))
        self.screen.blit(text, text_rect)

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
        self._render_animation_dashboard(state)
        self._render_plots_dashboard(state)

    def step(self) -> None:
        """Execute visualization step."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.screen.fill(self.background_color)
        state = self.resolved_inputs.find(self.inputs.inputs[0]).value

        if self.has_reward:
            reward = float(
                self.resolved_inputs.find(self.reward_variable.full_name).value
            )
            self.reward_history.append(reward)

        self._render_state(state)
        self._render_reward_dashboard()
        pygame.display.flip()
        self.clock.tick(self.fps)


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
        self._render_animation_dashboard(state)
        # Use parent's plot dashboard
        self._render_plots_dashboard(state)

    def _render_animation_dashboard(self, state: NumericArray) -> None:
        """Render pendulum animation in the left dashboard."""
        pygame.draw.rect(
            self.screen,
            (200, 200, 200),
            (0, 0, self.dashboard_width, self.dashboard_height),
            2,
        )

        center = (self.dashboard_width // 2, self.dashboard_height // 2)
        length = 200
        angle = state[0]

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
        # Override animation dashboard with point
        self._render_animation_dashboard(state)
        # Use parent's plot dashboard
        self._render_plots_dashboard(state)

    def _render_animation_dashboard(self, state: NumericArray) -> None:
        """Render point animation in the left dashboard."""
        pygame.draw.rect(
            self.screen,
            (200, 200, 200),
            (0, 0, self.dashboard_width, self.dashboard_height),
            2,
        )

        scale = 100
        x = self.dashboard_width // 2 + int(state[0] * scale)
        y = self.dashboard_height // 2 - int(state[1] * scale)

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

    def _render_animation_dashboard(self, state: NumericArray) -> None:
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
        x = self.dashboard_width // 2 + int(state[0] * scale)
        y = self.dashboard_height // 2 - int(state[1] * scale)
        angle = state[2]

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
