"""Base class for step modifiers."""

from abc import abstractmethod
from typing import Callable, Any, TypeVar
from regelum.environment.node.interfaces.modifiers import IStepModifier
from regelum.environment.node.interfaces.node import INode

T = TypeVar("T", bound=INode)


class StepModifier(IStepModifier[T]):
    """Abstract base class for step function modifiers."""

    _node: T
    _step_function: Callable[..., None]

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def step_function(self) -> Callable[..., None]:
        return self._step_function

    @abstractmethod
    def reset(self, *, apply_reset_modifier: bool = True) -> None:
        pass

    @property
    def node(self) -> T:
        return self._node

    def bind_to_node(self, node: T) -> None:
        """Bind the modifier to a node.

        This method:
        1. Stores the original step function
        2. Replaces it with the modified version
        3. Sets up reset behavior

        Args:
            node: Node to modify.
        """
        self._node = node
        original_reset = node.reset

        # Create bound method for step
        bound_call = self.__call__.__get__(self, self.__class__)
        node.step = bound_call  # type: ignore

        def modified_reset() -> None:
            """Modified reset that includes modifier reset."""
            original_reset(apply_reset_modifier=True)
            self.reset(apply_reset_modifier=True)

        node.modified_reset = modified_reset
        node.original_reset = original_reset
