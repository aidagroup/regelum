"""Core interfaces for the node-based computation system.

Includes variables, resolution, and reset capabilities.
"""

from __future__ import annotations
from abc import abstractmethod
from typing import (
    Any,
    Optional,
    List,
    Set,
    Tuple,
    TypeVar,
    Generic,
    Protocol,
    runtime_checkable,
)

from regelum.node.core.types import Metadata

T = TypeVar("T")
V = TypeVar("V")
T_co = TypeVar("T_co", covariant=True)
V_co = TypeVar("V_co", covariant=True)


@runtime_checkable
class IResettable(Protocol):
    """Interface for objects that can be reset to their initial state."""

    @abstractmethod
    def reset(self, *, apply_reset_modifier: bool = True) -> None:
        """Reset the object to its initial state.

        Args:
            apply_reset_modifier: Whether to apply reset modifier if available.
        """


@runtime_checkable
class IVariable(IResettable, Protocol):
    """Interface for variables in the computation graph."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Get variable name.

        Returns:
            Name of the variable.
        """

    @name.setter
    @abstractmethod
    def name(self, new_name: str) -> None:
        """Set variable name.

        Args:
            new_name: New name for the variable.
        """

    @property
    @abstractmethod
    def metadata(self) -> Metadata:
        """Get variable metadata.

        Returns:
            Metadata of the variable.
        """

    @property
    @abstractmethod
    def value(self) -> Any:
        """Get the current value of the variable.

        Returns:
            The current value.
        """

    @value.setter
    @abstractmethod
    def value(self, val: Any) -> None:
        """Set the current value of the variable.

        Args:
            val: The new value to set.
        """

    @property
    @abstractmethod
    def node_name(self) -> str:
        """Get name of the owning node.

        Returns:
            Name of the node that owns this variable.
        """

    @node_name.setter
    @abstractmethod
    def node_name(self, new_name: str) -> None:
        """Set name of the owning node.

        Args:
            new_name: New name for the node.
        """

    @abstractmethod
    def get_value(self) -> Any:
        """Get the current value of the variable.

        Returns:
            The current value, which could be symbolic or actual based on context.
        """

    @property
    @abstractmethod
    def full_name(self) -> str:
        """Get the fully qualified name of the variable.

        Returns:
            String in format 'node_name.variable_name'.
        """

    @abstractmethod
    def set_new_value(self, value: Any) -> None:
        """Set a new value and update initial value.

        Args:
            value: The new value to set.
        """


@runtime_checkable
class IResolvable(Protocol, Generic[T_co, V]):
    """Interface for objects that can resolve dependencies.

    Type Parameters:
        T_co: The type of the resolved result (covariant).
        V: The type of variables used for resolution (invariant).
    """

    @abstractmethod
    def resolve(self, variables: List[V]) -> Tuple[T_co, Set[str]]:
        """Resolve dependencies using provided variables.

        Args:
            variables: List of variables to resolve against.

        Returns:
            Tuple of (resolved result, set of unresolved dependencies).
        """


class IResolvedInputs(Protocol):
    """Interface for resolved input collections."""

    @abstractmethod
    def find(self, full_name: str) -> Optional[IVariable]:
        """Find a variable by its full name.

        Args:
            full_name: Fully qualified name to search for.

        Returns:
            Matching variable or None if not found.
        """

    @abstractmethod
    def __len__(self) -> int:
        """Get number of resolved inputs.

        Returns:
            Number of resolved inputs.
        """

    @property
    @abstractmethod
    def inputs(self) -> List[IVariable]:
        """Get list of resolved input variables.

        Returns:
            List of resolved variables.
        """


class IInputs(IResolvable[IResolvedInputs, IVariable], Protocol):
    """Interface for input collections."""

    @property
    @abstractmethod
    def inputs(self) -> List[str]:
        """Get list of input names.

        Returns:
            List of fully qualified input names.
        """
