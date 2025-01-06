"""Interface for nodes that form the basic building blocks of the computation graph."""

from __future__ import annotations
from abc import abstractmethod
from typing import (
    Optional,
    List,
    Dict,
    Any,
    Protocol,
    Sequence,
    Tuple,
    Set,
    Callable,
)
from regelum.environment.node.interfaces.base import (
    IVariable,
    IInputs,
    IResettable,
    IResolvable,
    IResolvedInputs,
)


class INode(IResettable, IResolvable[IInputs, IVariable], Protocol):
    """Interface for computational nodes in the graph.

    A node represents a unit of computation that can:
    - Execute a computational step
    - Reset its state
    - Resolve its dependencies
    - Manage variables and inputs
    """

    @abstractmethod
    def step(self) -> None:
        """Execute one computational step."""

    @property
    @abstractmethod
    def variables(self) -> Sequence[IVariable]:
        """Get list of variables owned by this node.

        Returns:
            List of variables defined in this node.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get internal name of the node.

        Returns:
            Internal name used for self-reference.
        """

    @property
    @abstractmethod
    def external_name(self) -> str:
        """Get external name of the node.

        Returns:
            Unique external name for identification.
        """

    @property
    @abstractmethod
    def is_resolved(self) -> bool:
        """Check if node's inputs are resolved.

        Returns:
            True if all inputs are resolved to variables.
        """

    @abstractmethod
    def get_full_names(self) -> List[str]:
        """Get fully qualified names of all variables.

        Returns:
            List of strings in format 'node_name.variable_name'.
        """

    @abstractmethod
    def alter_name(self, new_name: str) -> str:
        """Update node name and propagate to variables.

        Args:
            new_name: New name for the node.

        Returns:
            Previous external name.
        """

    @abstractmethod
    def define_variable(
        self,
        name: str,
        value: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        shape: Optional[tuple[int, ...]] = None,
    ) -> IVariable:
        """Create and register a new variable.

        Args:
            name: Variable name.
            value: Initial value.
            metadata: Optional metadata dictionary.
            shape: Optional shape tuple.

        Returns:
            Created variable instance.
        """

    @property
    @abstractmethod
    def step_size(self) -> Optional[float]:
        """Get node's execution time step.

        Returns:
            Time step for execution, or None if not time-dependent.
        """

    @step_size.setter
    @abstractmethod
    def step_size(self, value: Optional[float]) -> None:
        """Set node's execution time step.

        Args:
            value: Time step for execution, or None if not time-dependent.
        """

    @property
    @abstractmethod
    def is_continuous(self) -> bool:
        """Check if node represents continuous dynamics.

        Returns:
            True if node represents continuous dynamics.
        """

    @property
    @abstractmethod
    def is_root(self) -> bool:
        """Check if node is a root node.

        Returns:
            True if node is a root node.
        """

    @abstractmethod
    def find_variable(self, name: str) -> Optional[IVariable]:
        """Find variable by name.

        Args:
            name: Variable name to find.

        Returns:
            Variable if found, None otherwise.
        """

    @abstractmethod
    def get_variable(self, name: str) -> IVariable:
        """Get variable by name or raise error.

        Args:
            name: Variable name to get.

        Returns:
            Variable instance.

        Raises:
            ValueError: If variable not found.
        """

    @abstractmethod
    def alter_input_names(self, mapping: Dict[str, str]) -> None:
        """Update input names using mapping.

        Args:
            mapping: Dictionary mapping old names to new names.
        """

    @abstractmethod
    def alter_variable_names(self, mapping: Dict[str, str]) -> None:
        """Update variable names using mapping.

        Args:
            mapping: Dictionary mapping old names to new names.
        """

    @property
    @abstractmethod
    def original_reset(self) -> Optional[Callable[..., None]]:
        """Get the original reset function before any modifiers.

        Returns:
            The original reset function if available.
        """

    @original_reset.setter
    @abstractmethod
    def original_reset(self, value: Optional[Callable[..., None]]) -> None:
        """Set the original reset function.

        Args:
            value: The reset function to set as original.
        """

    @property
    @abstractmethod
    def modified_reset(self) -> Optional[Callable[..., None]]:
        """Get the modified reset function if available.

        Returns:
            The modified reset function if available.
        """

    @modified_reset.setter
    @abstractmethod
    def modified_reset(self, value: Optional[Callable[..., None]]) -> None:
        """Set the modified reset function.

        Args:
            value: The reset function to set as modified.
        """

    @property
    @abstractmethod
    def inputs(self) -> IInputs:
        """Get node's input dependencies.

        Returns:
            Input dependencies configuration.
        """

    @inputs.setter
    @abstractmethod
    def inputs(self, value: IInputs) -> None:
        """Set node's input dependencies.

        Args:
            value: Input dependencies configuration.
        """

    @property
    @abstractmethod
    def resolved_inputs(self) -> Optional[IResolvedInputs]:
        """Get node's resolved inputs.

        Returns:
            Resolved inputs configuration.
        """

    @resolved_inputs.setter
    @abstractmethod
    def resolved_inputs(self, value: Optional[IResolvedInputs]) -> None:
        """Set node's resolved inputs.

        Args:
            value: Resolved inputs configuration.
        """

    @classmethod
    @abstractmethod
    def get_instances(cls) -> List[INode]:
        """Get all instances of the node.

        Returns:
            List of node instances.
        """

    @classmethod
    @abstractmethod
    def get_instance_count(cls) -> int:
        """Get the number of instances of the node.

        Returns:
            Number of node instances.
        """

    @abstractmethod
    def get_resolved_inputs(
        self, variables: List[IVariable]
    ) -> Tuple[IResolvedInputs, Set[str]]:
        """Get node's resolved inputs.

        Returns:
            Resolved inputs configuration.
        """

    @property
    @abstractmethod
    def unresolved_inputs(self) -> Set[str]:
        """Get node's unresolved inputs.

        Returns:
            Set of unresolved input names.
        """

    @abstractmethod
    def __deepcopy__(self, memo: Dict[Any, Any]) -> "INode":
        """Deepcopy the node.

        This intended to be an abstract method as deepcopying the node is an ambiguous
        operation in the context of the graph.
        """
