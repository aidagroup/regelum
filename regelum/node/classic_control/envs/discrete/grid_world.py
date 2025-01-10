"""Grid world environment with discrete states and actions."""

from typing import Tuple, Optional, List
import numpy as np
from regelum.node.base import Node


class GridWorld(Node):
    """Grid world environment with discrete states and actions.

    A simple grid world where an agent can move in four directions.
    The state space is discrete, representing the agent's position.
    The action space is discrete with four possible actions.

    State:
        x: Integer grid position [row, col]

    Actions:
        0: Up
        1: Right
        2: Down
        3: Left

    Parameters:
        grid_size: Size of the grid (rows, cols)
        obstacles: List of obstacle positions [(row, col)]
        goal: Goal position (row, col)
    """

    def __init__(
        self,
        control_signal_name: str,
        grid_size: Tuple[int, int] = (5, 5),
        initial_state: Optional[np.ndarray] = None,
        obstacles: Optional[List[Tuple[int, int]]] = None,
        goal: Tuple[int, int] = (4, 4),
        step_size: float = 1.0,
    ):
        """Initialize grid world.

        Args:
            control_signal_name: Name of control input
            grid_size: Grid dimensions (rows, cols)
            initial_state: Initial position [row, col]
            obstacles: List of obstacle positions
            goal: Goal position
            step_size: Time step (discrete)
        """
        super().__init__(
            inputs=[control_signal_name],
            step_size=step_size,
            is_continuous=False,
            name="grid_world",
            is_root=True,
        )

        self.grid_size = grid_size
        self.obstacles = obstacles or []
        self.goal = goal

        if initial_state is None:
            initial_state = np.array([0, 0])

        assert 0 <= initial_state[0] < grid_size[0]
        assert 0 <= initial_state[1] < grid_size[1]
        assert tuple(initial_state) not in self.obstacles

        self.state = self.define_variable(
            "state",
            value=initial_state,
            metadata={"shape": (2,)},
        )

        # Action mappings
        self.actions = {
            0: np.array([-1, 0]),  # Up
            1: np.array([0, 1]),  # Right
            2: np.array([1, 0]),  # Down
            3: np.array([0, -1]),  # Left
        }

    def is_valid_position(self, position: np.ndarray) -> bool:
        """Check if position is valid (in bounds and not obstacle)."""
        row, col = position
        in_bounds = 0 <= row < self.grid_size[0] and 0 <= col < self.grid_size[1]
        return in_bounds and tuple(position) not in self.obstacles

    def step(self) -> None:
        """Execute one step in the environment."""
        if self.resolved_inputs is None:
            return

        action = self.resolved_inputs.find(self.inputs.inputs[0]).value
        action = int(action)  # Convert to integer index

        # Get action direction
        direction = self.actions[action]

        # Compute new position
        new_position = self.state.value + direction

        # Update state if valid move
        if self.is_valid_position(new_position):
            self.state.value = new_position

    def get_reward(self) -> float:
        """Compute reward for current state."""
        if tuple(self.state.value) == self.goal:
            return 1.0
        return 0.0

    def is_terminal(self) -> bool:
        """Check if current state is terminal."""
        return tuple(self.state.value) == self.goal
