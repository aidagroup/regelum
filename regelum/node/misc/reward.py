"""Reward tracker node implementation module."""

from abc import ABC, abstractmethod

from regelum.node.base import Node
from regelum import Variable
from regelum.utils import NumericArray


class RewardTracker(Node, ABC):
    """Node to track reward."""

    def __init__(self, state_variable: Variable):
        """Initialize reward tracker.

        Args:
            state_variable: State variable to track.
        """
        super().__init__(
            inputs=[state_variable.full_name],
            name=self.name,
            is_continuous=False,
            step_size=0,
        )
        self.state_variable = state_variable
        self.reward = self.define_variable(
            "reward",
            value=0.0,
            shape=(1,),
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the reward tracker."""

    @abstractmethod
    def objective_function(self, x: NumericArray) -> float:
        """Objective function to compute reward."""

    def step(self) -> None:
        self.reward.value = self.objective_function(
            self.resolved_inputs.find(self.state_variable.full_name).value
        )
