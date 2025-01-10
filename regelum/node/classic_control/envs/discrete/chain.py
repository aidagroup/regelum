"""Chain environment with discrete states and actions."""

from typing import Optional
import numpy as np
from regelum.node.base import Node


class Chain(Node):
    """Chain environment with discrete states and actions.

    A simple 1D chain where an agent can move left or right.
    The state space is discrete integers from 0 to length-1.
    The action space is discrete with two possible actions.

    State:
        x: Integer position in chain [0, length-1]

    Actions:
        0: Left
        1: Right

    Parameters:
        length: Length of the chain
        goal: Goal position
    """

    def __init__(
        self,
        control_signal_name: str,
        length: int = 10,
        initial_state: Optional[np.ndarray] = None,
        goal: int = 9,
        step_size: float = 1.0,
    ):
        """Initialize chain environment.

        Args:
            control_signal_name: Name of control input
            length: Length of the chain
            initial_state: Initial position
            goal: Goal position
            step_size: Time step (discrete)
        """
        super().__init__(
            inputs=[control_signal_name],
            step_size=step_size,
            is_continuous=False,
            name="chain",
            is_root=True,
        )

        self.length = length
        self.goal = goal

        if initial_state is None:
            initial_state = np.array([0])

        assert 0 <= initial_state[0] < length
        assert 0 <= goal < length

        self.state = self.define_variable(
            "state",
            value=initial_state,
            metadata={"shape": (1,)},
        )

        self.actions = {
            0: -1,  # Left
            1: 1,  # Right
        }

    def step(self) -> None:
        """Execute one step in the environment."""
        action = int(self.resolved_inputs.inputs[0].value)
        direction = self.actions[action]

        # Compute new position
        new_position = self.state.value + direction

        # Update state if valid move
        if 0 <= new_position < self.length:
            self.state.value = new_position

    def get_reward(self) -> float:
        """Compute reward for current state."""
        if self.state.value[0] == self.goal:
            return 1.0
        return 0.0
