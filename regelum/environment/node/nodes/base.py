"""Base node implementation module."""

from abc import abstractmethod
from typing import (
    Dict,
    List,
    Optional,
    Any,
    ClassVar,
    Sequence,
    Tuple,
    Set,
    Callable,
    Self,
    cast,
    TypeVar,
    Type,
    Union,
)
from copy import deepcopy

from regelum.environment.node.interfaces.node import INode
from regelum.environment.node.interfaces.base import IVariable, IInputs, IResolvedInputs
from regelum.environment.node.core.variable import Variable
from regelum.environment.node.core.inputs import Inputs, ResolvedInputs
from regelum.environment.node.core.types import (
    MetadataKey,
    default_metadata,
    Value,
    Shape,
)


class Node(INode):
    """Base implementation of INode for computational nodes.

    A node is a unit of computation that:
    - Manages its own variables
    - Can be reset to initial state
    - Resolves dependencies on other nodes
    - Executes computational steps
    """

    _instances: ClassVar[Dict[str, List["Node"]]] = {}
    _inputs: IInputs
    _step_size: Optional[float]
    _is_continuous: bool
    _is_root: bool
    _variables: List[IVariable]
    _resolved_inputs: Optional[ResolvedInputs]
    _internal_name: str
    _external_name: str
    _reset_with_modifier: Optional[Callable[[Optional[List[str]], bool], None]]
    last_update_time: Optional[float]
    _original_reset: Optional[Callable[..., None]] = None
    _modified_reset: Optional[Callable[..., None]] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        """Create new node instance and track it.

        Args:
            *args: Positional arguments for __init__.
            **kwargs: Keyword arguments for __init__.

        Returns:
            New Node instance with proper type inference.
        """
        instance = super().__new__(cls)
        cls._instances.setdefault(cls.__name__, []).append(instance)
        return instance

    def __init__(
        self,
        inputs: Optional[Inputs | List[str]] = None,
        *,
        step_size: Optional[float] = None,
        is_continuous: bool = False,
        is_root: bool = False,
        name: Optional[str] = None,
    ) -> None:
        """Initialize node with configuration.

        Args:
            inputs: Input dependencies.
            step_size: Time step for execution.
            is_continuous: Whether node represents continuous dynamics.
            is_root: Whether node is a root node.
            name: Optional name override.
        """
        self._inputs = self._normalize_inputs(inputs)
        self._step_size = step_size
        self._is_continuous = is_continuous
        self._is_root = is_root
        self._variables = []
        self._resolved_inputs: Optional[ResolvedInputs] = None
        self._original_reset: Optional[Callable[..., None]] = None
        self._modified_reset: Optional[Callable[..., None]] = None

        if is_continuous and not hasattr(self, "state_transition_map"):
            raise ValueError(
                f"Continuous node {self.__class__.__name__} must implement state_transition_map"
            )

        self._internal_name = name or self.__class__.__name__.lower()
        self._external_name = f"{self._internal_name}_{self.get_instance_count()}"
        self._reset_with_modifier = None

    def _normalize_inputs(self, inputs: Optional[IInputs | List[str]]) -> IInputs:
        """Convert inputs to Inputs instance.

        Args:
            inputs: Input specification.

        Returns:
            Normalized Inputs instance.
        """
        if not hasattr(self, "_inputs"):
            if inputs is None:
                return Inputs([])
            return Inputs(inputs) if isinstance(inputs, list) else inputs
        return Inputs(self._inputs) if isinstance(self._inputs, list) else self._inputs

    @abstractmethod
    def step(self) -> None:
        """Execute one computational step."""
        pass

    @property
    def variables(self) -> Sequence[IVariable]:
        """Get list of node variables.

        Returns:
            List of variables defined in this node.
        """
        return self._variables

    @property
    def name(self) -> str:
        """Get internal name.

        Returns:
            Internal name used for self-reference.
        """
        return self._internal_name

    @property
    def external_name(self) -> str:
        """Get external name.

        Returns:
            Unique external name for identification.
        """
        return self._external_name

    @property
    def is_resolved(self) -> bool:
        """Check if inputs are resolved.

        Returns:
            True if all inputs are resolved to variables.
        """
        return self._resolved_inputs is not None

    def get_full_names(self) -> List[str]:
        """Get fully qualified names of all variables.

        Returns:
            List of strings in format 'node_name.variable_name'.
        """
        return [f"{self.external_name}.{var.name}" for var in self._variables]

    def alter_name(self, new_name: str) -> str:
        """Update node name and propagate to variables.

        Args:
            new_name: New name for the node.

        Returns:
            Previous external name.
        """
        old_name = self.external_name
        self._internal_name = new_name
        self._external_name = f"{new_name}_{self.get_instance_count()}"
        for var in self._variables:
            var.node_name = self.external_name
        return old_name

    def define_variable(
        self,
        name: str,
        value: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None,
        shape: Optional[tuple[int, ...]] = None,
        reset_modifier: Optional[Callable[[Any], Any]] = None,
    ) -> IVariable:
        """Create and register a new variable."""
        base_meta = default_metadata()
        if metadata:
            typed_meta: Dict[MetadataKey, Union[Value, Shape, Callable[[Any], Any]]] = {
                cast(MetadataKey, k): v
                for k, v in metadata.items()
                if k in MetadataKey.__args__  # type: ignore
            }
            base_meta.update(typed_meta)
        if shape is not None:
            base_meta["shape"] = shape
        base_meta["current_value"] = value
        base_meta["reset_modifier"] = reset_modifier
        var = Variable(name=name, metadata=base_meta, _node_name=self.external_name)
        self._variables.append(var)
        return var

    T = TypeVar("T", bound="Node")

    @property
    def step_size(self) -> Optional[float]:
        """Get step size."""
        return self._step_size

    @step_size.setter
    def step_size(self, value: Optional[float]) -> None:
        """Set step size."""
        self._step_size = value

    @property
    def is_continuous(self) -> bool:
        """Get continuous flag."""
        return self._is_continuous

    @property
    def is_root(self) -> bool:
        """Get root flag."""
        return self._is_root

    def find_variable(self, name: str) -> Optional[IVariable]:
        """Find variable by name."""
        return next((var for var in self._variables if var.name == name), None)

    def get_variable(self, name: str) -> IVariable:
        """Get variable by name or raise error."""
        if var := self.find_variable(name):
            return var
        raise ValueError(f"Variable '{name}' not found in node '{self.external_name}'")

    def alter_input_names(self, mapping: Dict[str, str]) -> None:
        """Update input names using mapping."""
        self._inputs = Inputs([mapping.get(name, name) for name in self._inputs.inputs])

    def alter_variable_names(self, mapping: Dict[str, str]) -> None:
        """Update variable names using mapping."""
        for var in self._variables:
            var.name = mapping.get(var.name, var.name)

    def reset(self, *, apply_reset_modifier: bool = True) -> None:
        """Reset the node to its initial state.

        Args:
            apply_reset_modifier: Whether to apply reset modifier if available.
        """
        if apply_reset_modifier and self._modified_reset is not None:
            self._modified_reset()
        else:
            self._reset()

    def _reset(self, variables_to_reset: Optional[List[str]] = None) -> None:
        """Internal reset implementation.

        Args:
            variables_to_reset: Optional list of variable names to reset. If None, resets all variables.
        """
        if variables_to_reset is None:
            variables_to_reset = [var.name for var in self._variables]
        for var_name in variables_to_reset:
            var = self.get_variable(var_name)
            var.reset()

    @property
    def original_reset(self) -> Optional[Callable[..., None]]:
        """Get the original reset function before any modifiers."""
        return self._original_reset or self._reset

    @original_reset.setter
    def original_reset(self, value: Optional[Callable[..., None]]) -> None:
        """Set the original reset function.

        Args:
            value: The reset function to set as original.
        """
        self._original_reset = value

    @property
    def modified_reset(self) -> Optional[Callable[..., None]]:
        """Get the modified reset function if available."""
        return self._modified_reset

    @modified_reset.setter
    def modified_reset(self, value: Optional[Callable[..., None]]) -> None:
        """Set the modified reset function.

        Args:
            value: The reset function to set as modified.
        """
        self._modified_reset = value

    @property
    def inputs(self) -> IInputs:
        """Get node's input dependencies."""
        return self._inputs

    @inputs.setter
    def inputs(self, value: Inputs) -> None:
        """Set inputs."""
        self._inputs = value

    @property
    def resolved_inputs(self) -> Optional[ResolvedInputs]:
        """Get resolved inputs."""
        return self._resolved_inputs

    @resolved_inputs.setter
    def resolved_inputs(self, value: Optional[ResolvedInputs]) -> None:
        """Set resolved inputs."""
        self._resolved_inputs = value

    @classmethod
    def get_instances(cls: Type[T]) -> List[INode]:
        """Get all instances of this class.

        Returns:
            List of all instances created.
        """
        return cast(List[INode], cls._instances.get(cls.__name__, []))

    @classmethod
    def get_instance_count(cls) -> int:
        """Get count of instances for this class.

        Returns:
            Number of instances created.
        """
        return len(cls._instances.get(cls.__name__, []))

    def get_resolved_inputs(
        self, variables: List[IVariable]
    ) -> Tuple[IResolvedInputs, Set[str]]:
        """Get resolved inputs and unresolved names."""
        return self._inputs.resolve(variables)

    @property
    def unresolved_inputs(self) -> Set[str]:
        """Get unresolved input names."""
        if not self._resolved_inputs:
            _, unresolved = self._inputs.resolve([])
            return unresolved
        return set()

    def __str__(self) -> str:
        """String representation."""
        class_repr = f"{self.__class__.__name__}({self.external_name})"
        return f"{class_repr}, inputs={self._inputs}, variables={self._variables})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()

    def resolve(self, variables: List[IVariable]) -> Tuple[IInputs, Set[str]]:
        """Resolve node inputs."""
        resolved, unresolved = self._inputs.resolve(variables)
        self._resolved_inputs = (
            resolved if isinstance(resolved, ResolvedInputs) else None
        )
        if unresolved:
            raise ValueError(
                f"Couldn't resolve inputs {unresolved} for node {self.external_name}"
            )
        if not isinstance(resolved, ResolvedInputs):
            raise ValueError("Failed to resolve inputs to ResolvedInputs type")
        return self.inputs, unresolved

    def __deepcopy__(self, memo: Dict[int, Any]) -> Self:
        """Create a deep copy of the node.

        Args:
            memo: Dictionary of already copied objects.

        Returns:
            New Node instance with copied data.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        result._internal_name = self._internal_name
        result._external_name = (
            f"{result._internal_name}_{len(cls._instances[cls.__name__])}"
        )

        for k, v in self.__dict__.items():
            if k == "_variables":
                vars_copy: List[IVariable] = [deepcopy(var, memo) for var in v]
                for var in vars_copy:
                    var.node_name = result._external_name
                setattr(result, k, vars_copy)
            elif k == "_resolved_inputs":
                setattr(result, k, None)
            else:
                if k != "_external_name":
                    setattr(result, k, deepcopy(v, memo))

        return result
