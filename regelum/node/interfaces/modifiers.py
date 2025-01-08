"""Interfaces for node behavior modifiers.

Includes step control, reset handling, and state management.
"""

from abc import abstractmethod
from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)
from .node import INode
from .base import IResettable

T = TypeVar("T", bound=INode)
V_co = TypeVar("V_co", covariant=True)


@runtime_checkable
class IModifier(Protocol, Generic[T, V_co]):
    """Interface for node modifiers.

    A modifier alters the behavior of a node by wrapping its methods.
    Common use cases include:
    - Adding timing control (ZeroOrderHold)
    - Adding reset behavior
    - Adding state persistence
    - Adding logging or monitoring

    Type Parameters:
        T: Type of node being modified, must implement INode.
        V: Type of value being modified/processed.
    """

    @property
    @abstractmethod
    def node(self) -> T:
        """Get the node being modified.

        Returns:
            The wrapped node instance.
        """

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> V_co:
        """Execute the modified behavior.

        This method wraps the original node method and adds the modifier's behavior.

        Args:
            *args: Positional arguments to pass to the original method.
            **kwargs: Keyword arguments to pass to the original method.

        Returns:
            Modified result of the original method call.
        """

    @abstractmethod
    def bind_to_node(self, node: T) -> None:
        """Bind this modifier to a node.

        This method should:
        1. Store the original method(s) being modified
        2. Replace the method(s) with the modified version(s)
        3. Set up any necessary state

        Args:
            node: Node to modify.
        """


@runtime_checkable
class IStepModifier(IModifier[T, None], IResettable, Protocol):
    """Interface for modifiers that alter the step behavior of a node.

    Step modifiers specifically target the step() method of nodes and can:
    - Control when steps are executed
    - Add pre/post step processing
    - Modify step frequency
    - Add step-specific state
    """

    @property
    @abstractmethod
    def step_function(self) -> Callable[..., None]:
        """Get the original step function being modified.

        Returns:
            The original unmodified step function.
        """


@runtime_checkable
class IResetModifier(IModifier[T, None], Protocol):
    """Interface for modifiers that alter the reset behavior of a node.

    Reset modifiers specifically target the reset() method of nodes and can:
    - Add custom reset logic
    - Control what gets reset
    - Add reset-specific state
    - Handle reset propagation
    """

    @property
    @abstractmethod
    def reset_semaphore(self) -> Optional[INode]:
        """Get the reset semaphore if one is being used.

        Returns:
            The reset semaphore node if one exists, None otherwise.
        """
