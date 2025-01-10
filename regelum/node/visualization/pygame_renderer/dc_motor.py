"""PyGame visualization node for DC motor system."""

from typing import Optional

import pygame
import numpy as np

from regelum.node.visualization.pygame_renderer import PyGameRenderer
from regelum import Variable
from regelum.node.core.types import NumericArray


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
        record_video: bool = False,
        video_path: str = "my_animation.avi",
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
            record_video: Whether to record a video of the simulation
            video_path: Path to save the video
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
            record_video=record_video,
            video_path=video_path,
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
        """Render a single motor visualization."""
        theta, _, current = state[0], state[1], state[2]
        housing_color, rotor_color, indicator_color = color_scheme

        # Draw motor housing with gradient effect
        for i in range(4):
            shade = max(0, housing_color[0] - i * 15)
            pygame.draw.circle(
                self.screen,
                (shade, shade, shade),
                (center_x, center_y),
                motor_radius - i,
            )

        # Inner motor circle with metallic look
        pygame.draw.circle(
            self.screen,
            (180, 180, 180),
            (center_x, center_y),
            motor_radius - 8,
        )
        pygame.draw.circle(
            self.screen,
            (200, 200, 200),
            (center_x, center_y),
            motor_radius - 12,
        )

        # Draw rotor with thickness and shading
        indicator_length = motor_radius - 16
        end_x = center_x + indicator_length * np.cos(theta)
        end_y = center_y + indicator_length * np.sin(theta)

        # Rotor shadow
        pygame.draw.line(
            self.screen,
            (50, 50, 50),
            (center_x + 2, center_y + 2),
            (end_x + 2, end_y + 2),
            8,
        )

        # Main rotor
        pygame.draw.line(
            self.screen,
            rotor_color,
            (center_x, center_y),
            (end_x, end_y),
            6,
        )

        # Draw mounting brackets with 3D effect
        bracket_width = 24
        bracket_height = 16
        for y_offset in [-motor_radius - 5, motor_radius - bracket_height + 5]:
            # Bracket shadow
            pygame.draw.rect(
                self.screen,
                (100, 100, 100),
                (
                    center_x - bracket_width // 2 + 2,
                    center_y + y_offset + 2,
                    bracket_width,
                    bracket_height,
                ),
            )
            # Main bracket
            pygame.draw.rect(
                self.screen,
                indicator_color,
                (
                    center_x - bracket_width // 2,
                    center_y + y_offset,
                    bracket_width,
                    bracket_height,
                ),
            )

        # Draw center bearing
        pygame.draw.circle(
            self.screen,
            (100, 100, 100),
            (center_x, center_y),
            8,
        )
        pygame.draw.circle(
            self.screen,
            (150, 150, 150),
            (center_x, center_y),
            6,
        )

        # Current flow indicator
        if self.show_current_flow and abs(current) > 0.1:
            flow_points = []
            for i in range(8):
                angle = 2 * np.pi * i / 8 + self.current_flow_offset
                x = center_x + (motor_radius + 15) * np.cos(angle)
                y = center_y + (motor_radius + 15) * np.sin(angle)
                flow_points.append((x, y))

            current_color = (0, 0, 255) if current > 0 else (255, 0, 0)
            for i in range(len(flow_points)):
                start = flow_points[i]
                end = flow_points[(i + 1) % len(flow_points)]
                pygame.draw.line(self.screen, current_color, start, end, 2)

            # Update flow animation
            self.current_flow_offset += 0.1 * np.sign(current)

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

        true_center_x = self.dashboard_width // 2
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
            est_center_x = 3 * self.dashboard_width // 4
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
