"""PyGame visualization node for system animation."""

from typing import Optional, List
import pygame
import numpy as np
from regelum import Node
from regelum.node.core.variable import Variable
from regelum.node.core.types import NumericArray


class PyGameRenderer(Node):
    """Base PyGame renderer for system visualization."""

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

            mid_y = plot_y + plot_area_height // 2
            pygame.draw.line(
                self.screen,
                (100, 100, 100),
                (self.dashboard_width + 50, mid_y),
                (self.window_size[0] - 50, mid_y),
                1,
            )

            y_label_min = self.font.render(f"{y_min:.2f}", True, (0, 0, 0))
            y_label_max = self.font.render(f"{y_max:.2f}", True, (0, 0, 0))
            self.screen.blit(
                y_label_min, (self.dashboard_width + 10, plot_y + plot_area_height - 10)
            )
            self.screen.blit(y_label_max, (self.dashboard_width + 10, plot_y))

            x_min = max(0, self.current_step - self.visible_history)
            x_max = self.current_step
            x_label_min = self.font.render(f"{x_min}", True, (0, 0, 0))
            x_label_max = self.font.render(f"{x_max}", True, (0, 0, 0))
            self.screen.blit(
                x_label_min,
                (self.dashboard_width + 45, plot_y + plot_area_height + 5),
            )
            self.screen.blit(
                x_label_max,
                (self.window_size[0] - 60, plot_y + plot_area_height + 5),
            )

            text = self.font.render(f"State {i}", True, (0, 0, 0))
            text_rect = text.get_rect(midright=(self.dashboard_width + 45, mid_y))
            self.screen.blit(text, text_rect)

            points = []
            for t, value in enumerate(visible_history):
                x = (
                    self.dashboard_width
                    + 50
                    + (t * plot_width) // min(self.visible_history, len(history))
                )
                normalized_y = (value - y_min) / (y_max - y_min)
                y = plot_y + plot_area_height - int(normalized_y * plot_area_height)
                y = max(plot_y, min(y, plot_y + plot_area_height))
                points.append((x, y))

            if len(points) > 1:
                pygame.draw.lines(self.screen, (0, 100, 200), False, points, 2)

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

        pygame.draw.rect(
            self.screen,
            (240, 240, 240),
            (x_start + 50, margin, plot_width, plot_height),
        )

        mid_y = margin + plot_height // 2
        pygame.draw.line(
            self.screen,
            (100, 100, 100),
            (x_start + 50, mid_y),
            (x_start + 50 + plot_width, mid_y),
            1,
        )

        y_label_min = self.font.render(f"{y_min:.2f}", True, (0, 0, 0))
        y_label_max = self.font.render(f"{y_max:.2f}", True, (0, 0, 0))
        self.screen.blit(y_label_min, (x_start + 10, margin + plot_height - 10))
        self.screen.blit(y_label_max, (x_start + 10, margin))

        x_min = max(0, self.current_step - self.visible_history)
        x_max = self.current_step
        x_label_min = self.font.render(f"{x_min}", True, (0, 0, 0))
        x_label_max = self.font.render(f"{x_max}", True, (0, 0, 0))
        self.screen.blit(x_label_min, (x_start + 45, margin + plot_height + 5))
        self.screen.blit(
            x_label_max, (x_start + plot_width - 10, margin + plot_height + 5)
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
            pygame.draw.lines(self.screen, (255, 100, 100), False, points, 2)

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
    """PyGame renderer for pendulum system."""

    def _render_state(self, state: NumericArray) -> None:
        center = (self.window_size[0] // 2, self.window_size[1] // 2)
        length = 200
        angle = state[0]

        end_pos = (
            center[0] + length * np.sin(angle),
            center[1] + length * np.cos(angle),
        )

        pygame.draw.circle(self.screen, (0, 0, 0), center, 10)
        pygame.draw.line(self.screen, (0, 0, 0), center, end_pos, 4)
        pygame.draw.circle(
            self.screen, (255, 0, 0), (int(end_pos[0]), int(end_pos[1])), 20
        )


class KinematicPointRenderer(PyGameRenderer):
    """PyGame renderer for kinematic point system."""

    def _render_state(self, state: NumericArray) -> None:
        scale = 100
        x = self.window_size[0] // 2 + int(state[0] * scale)
        y = self.window_size[1] // 2 - int(state[1] * scale)

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
    """PyGame renderer for three wheeled robot system."""

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
