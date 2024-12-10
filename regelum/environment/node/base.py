"""Environment node module for building hierarchical state-based systems.

This module provides base classes for creating nodes in a computational graph,
managing state hierarchies, and implementing MPC controllers.

The module includes:
    - State: Wrapper for hierarchical state management
    - Inputs: Handler for node input dependencies
    - Node: Base class for all computational nodes
    - Graph: Manages node execution and dependencies
    - Clock: Provides timing functionality
    - Logger: Records state evolution
    - MPCNode: Implements Model Predictive Control
"""

from __future__ import annotations
from functools import reduce
from typing import (
    Dict,
    Optional,
    Tuple,
    Union,
    List,
    Any,
    Type,
    TypeVar,
    overload,
    Literal,
    cast,
    Callable,
)
from copy import deepcopy

try:
    from typing import TypeGuard  # Python 3.10+
except ImportError:
    from typing_extensions import TypeGuard  # Python < 3.10
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
import casadi as cs
import numpy as np
from regelum import _SYMBOLIC_INFERENCE_ACTIVE
import logging

# if TYPE_CHECKING:
from regelum.environment.transistor import Transistor
from regelum.typing import (
    RgArray,
)
from math import gcd
from regelum.environment.transistor import (
    ScipyTransistor,
    SampleAndHoldModifier,
)

T = TypeVar("T")


@dataclass
class State:
    """Hierarchical state container with path-based access.

    Args:
        name: State identifier
        shape: Shape of the state data
        _value: State data or list of child states
        is_leaf: Auto-determined leaf status
        _initial_value: Initial value of the state
        _reset_modifier: Optional modifier function for state reset
    """

    name: str
    shape: Optional[Tuple[int, ...]] = None
    _value: Union[Any, List["State"], None] = None
    is_leaf: bool = field(init=False)
    _initial_value: Union[Any, List["State"], None] = field(init=False)
    _reset_modifier: Optional[Callable[[Any], Any]] = field(default=None)

    def __post_init__(self):
        self.is_leaf = self._determine_leaf_status()
        self._validate_hierarchical_state()
        self._path_cache = {}
        self._build_path_cache()
        # Store initial value for reset
        self._initial_value = self._clone_value(self._value)

    def _determine_leaf_status(self) -> bool:
        return not (
            isinstance(self._value, list)
            and len(self._value) > 0
            and all(isinstance(s, State) for s in self._value)
        )

    def _validate_hierarchical_state(self):
        if not self.is_leaf and not all(
            isinstance(s, State) for s in self.get_value(is_leaf=False)
        ):
            raise TypeError(
                f"The _value of hierarchical State '{self.name}' must be a list of State instances."
            )

    def with_altered_name(self, new_name: str):
        self.name = new_name
        new_instance = deepcopy(self)
        new_instance._build_path_cache()
        return new_instance

    @property
    def value(self):
        """Return a dict representation of the state."""
        return (
            self._get_leaf_value() if self.is_leaf else self._get_hierarchical_value()
        )

    def _get_leaf_value(self):
        symbolic = getattr(_SYMBOLIC_INFERENCE_ACTIVE, "value", False)
        val = self.to_casadi_symbolic() if symbolic else self._value
        return {"name": self.name, "shape": self.shape, "value": val}

    def _get_hierarchical_value(self):
        return {
            "name": self.name,
            "shape": self.shape,
            "states": [substate.value for substate in self.get_value(is_leaf=False)],
        }

    @value.setter
    def value(self, new_value):
        self._value = new_value

    def to_casadi_symbolic(self) -> Optional[cs.MX]:
        """Convert the state to a CasADi symbolic object."""
        if not hasattr(self, "symbolic_value"):
            if self.shape:
                self.symbolic_value = cs.MX.sym(self.name, *self.shape)
            else:
                self.symbolic_value = cs.MX.sym(self.name)
        return self.symbolic_value

    def __getitem__(self, key: str):
        return self.search_by_path(key)

    def search_by_path(self, path: str) -> Optional["State"]:
        """Search for a substate by its path using cache."""
        return self._path_cache.get(path)

    @property
    def paths(self) -> List[str]:
        """Return all paths to leaf states."""
        paths = []
        self._collect_paths(prefix="", paths=paths)
        return paths

    def _collect_paths(self, prefix: str, paths: List[str]):
        """Helper method to collect paths recursively."""
        current_path = f"{prefix}/{self.name}" if prefix else self.name
        if self.is_leaf:
            paths.append(current_path)
        else:
            for substate in self.get_value(is_leaf=False):
                substate._collect_paths(prefix=current_path, paths=paths)

    def get_all_states(self) -> List["State"]:
        """Get a list of all leaf states."""
        states = []
        self._collect_states(states=states)
        return states

    def _collect_states(self, states: List["State"]):
        if self.is_leaf:
            states.append(self)
        else:
            states.append(self)
            for substate in self.get_value(is_leaf=False):
                substate._collect_states(states=states)

    @property
    def is_defined(self) -> bool:
        """Check if all leaf states have a defined value."""
        return all(state._value is not None for state in self.get_all_states())

    @property
    def data(self):
        """Direct accessor for state value data."""
        state_value = self.value
        if "value" in state_value:
            return state_value["value"]
        else:
            return state_value["states"][0]

    @data.setter
    def data(self, new_value):
        """Setter for state value data."""
        self._value = new_value
        self.is_leaf = self._determine_leaf_status()
        self._validate_hierarchical_state()
        self._build_path_cache()

    def get_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Get the shapes of all leaf states."""
        return {path: self[path].shape for path in self.paths}

    def _build_path_cache(self):
        """Build a cache of all possible paths to substates."""

        def _recurse(state: State, current_path: str):
            full_path = f"{current_path}/{state.name}" if current_path else state.name
            self._path_cache[full_path] = state

            if not state.is_leaf:
                for substate in state._value:
                    _recurse(substate, full_path)

        _recurse(self, "")

    @property
    def hierarchical_value(self) -> TypeGuard[List["State"]]:
        """Type guard to ensure _value is List[State] when not leaf."""
        return not self.is_leaf and isinstance(self._value, list)

    @overload
    def get_value(self: "State", *, is_leaf: Literal[True]) -> Any: ...

    @overload
    def get_value(self: "State", *, is_leaf: Literal[False]) -> List["State"]: ...

    def get_value(self, *, is_leaf: bool) -> Union[Any, List["State"]]:
        if is_leaf:
            return self._value
        return cast(List["State"], self._value)

    def _clone_value(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.copy()
        elif isinstance(value, list) and all(isinstance(x, State) for x in value):
            return [State(s.name, s.shape, self._clone_value(s._value)) for s in value]
        return value

    def reset(self) -> None:
        """Reset state to initial value, applying modifier if present."""
        initial = self._clone_value(self._initial_value)
        if self._reset_modifier and self.is_leaf:
            self._value = self._reset_modifier(initial)
        else:
            self._value = initial

        if not self.is_leaf:
            for substate in self.get_value(is_leaf=False):
                substate.reset()
            self._build_path_cache()

    def with_reset_modifier(self, modifier: Callable[[Any], Any]) -> "State":
        """Add a modifier function to transform state value during reset.

        Args:
            modifier: Function that takes current initial value and returns modified value
        """
        self._reset_modifier = modifier
        return self


@dataclass
class Inputs:
    """Input dependency manager for nodes.

    Args:
        paths_to_states: List of state paths required as inputs
        states: Resolved State objects
        _resolved: Resolution status flag
    """

    paths_to_states: List[str]
    states: List[State] = field(default_factory=list)
    _resolved: bool = False

    def resolve(self, states: List[State]):
        """Resolve the input paths to actual State instances."""
        found_states: List[State] = []

        for path in self.paths_to_states:
            found = next(
                (
                    state.search_by_path(path=path)
                    for state in states
                    if state.search_by_path(path=path)
                ),
                None,
            )
            if found:
                found_states.append(found)

        if len(found_states) != len(self.paths_to_states):
            missing = set([path.split("/")[-1] for path in self.paths_to_states]) - {
                state.name for state in found_states
            }
            raise ValueError(
                f"Could not resolve all input paths for {self.paths_to_states}. Missing: {missing}"
            )

        assert all(
            state.is_leaf for state in found_states
        ), "All inputs must be leaf states"
        self.states = found_states
        self._resolved = True

    def collect(self) -> Dict[str, Any]:
        """Collect the values of the input states, symbolic or numeric depending on context."""
        if len(self.paths_to_states) > 0:
            if not self._resolved:
                raise ValueError("Resolve inputs before collecting")
            return {
                path: state.data
                for path, state in zip(self.paths_to_states, self.states)
            }
        else:
            return {}

    def __getitem__(self, key: str) -> State:
        """Get a resolved input state by its name."""
        if not self._resolved:
            raise ValueError("Resolve inputs before accessing them")
        try:
            index = self.paths_to_states.index(key)
            return self.states[index]
        except ValueError as err:
            raise KeyError(f"Input '{key}' not found in paths_to_states") from err


class Node(ABC):
    """Base class for computational nodes.

    Args:
        is_root: Whether node is a root node
        step_size: Node's time step size
        state: Node's state object
        inputs: Required input state paths
    """

    def __init__(
        self,
        is_root: bool = False,
        step_size: Optional[float] = None,
        state: Optional[State] = None,
        inputs: Optional[List[str]] = None,
        is_continuous: bool = False,
        default_transistor_configuration: Optional[Dict[str, Any]] = None,
        prefix: Optional[str] = None,
    ) -> None:
        """Initialize the Node.

        Args:
            is_root: Whether node is a root node.
            step_size: Node's time step size.
            state: Node's state object.
            inputs: Required input state paths.
            is_continuous: Whether node represents continuous dynamics.
            default_transistor_configuration: Default configuration for transistor.
        """
        if not hasattr(self, "state"):
            if state is None:
                raise ValueError("State must be fully specified.")
            self.state = state
            if prefix is not None:
                self.state.name = f"{prefix}_{self.state.name}"
                self.state._build_path_cache()

        if not hasattr(self, "inputs"):
            if inputs is None:
                inputs = []
            self.inputs = Inputs(inputs) if isinstance(inputs, list) else inputs
        else:
            if isinstance(self.inputs, list):
                self.inputs = Inputs(self.inputs)
            elif isinstance(self.inputs, Inputs):
                pass
            else:
                raise ValueError("Inputs must be a list of strings")

        if self.state is None:
            raise ValueError("State must be fully specified.")
        self.is_continuous = is_continuous
        self.is_root = is_root
        self.step_size = step_size
        if self.is_root:
            assert (
                self.state.is_defined
            ), f"Initial state must be defined for the root node {self.state.name}"
        self.transistor = None
        if default_transistor_configuration is not None:
            self.default_transistor_configuration = default_transistor_configuration
        else:
            if self.is_continuous:
                self.default_transistor_configuration = {
                    "transistor": ScipyTransistor,
                    "transistor_kwargs": {},
                }
            else:
                self.default_transistor_configuration = {
                    "transistor": Transistor,
                    "transistor_kwargs": {},
                }

    def with_transistor(self, transistor: Type[Transistor], **transistor_kwargs):
        self.transistor = transistor(node=self, **transistor_kwargs)
        return self

    @abstractmethod
    def compute_state_dynamics(self) -> Dict[str, Any]:
        """Compute the state dynamics given inputs."""
        pass

    def reset(self, states_to_reset: Optional[List[str]] = None) -> None:
        """Reset node state to initial values.

        Args:
            states_to_reset: List of state paths to reset. If None, resets all states.
        """
        if states_to_reset is None:
            self.state.reset()
            return

        found_states = set()
        for path in states_to_reset:
            if state := self.state.search_by_path(path):
                state.reset()
                found_states.add(path)

        return found_states


class Graph:
    """Manages node execution order and dependencies.

    Args:
        nodes: List of nodes to manage
        states_to_log: State paths to record
        logger_cooldown: Minimum time between logs
    """

    def __init__(
        self,
        nodes: List[Node],
        states_to_log: Optional[List[str]] = None,
        logger_cooldown: float = 0.0,
    ) -> None:
        """Initialize the Graph.

        Args:
            nodes: List of nodes to manage.
            states_to_log: State paths to record.
            logger_cooldown: Minimum time between logs.
        """
        fundamental_step_size = self._validate_and_set_step_sizes(nodes)
        self._setup_logger(nodes, states_to_log, logger_cooldown, fundamental_step_size)
        self._initialize_graph(nodes, fundamental_step_size)

    def define_fundamental_step_size(self, nodes: List[Node]):
        step_sizes = [
            node.step_size
            for node in nodes
            if not node.is_continuous and node.step_size is not None
        ]

        def float_gcd(a: float, b: float) -> float:
            precision = 1e-9
            a, b = round(a / precision), round(b / precision)
            return gcd(int(a), int(b)) * precision

        fundamental_step_size = (
            reduce(float_gcd, step_sizes) if len(set(step_sizes)) > 1 else step_sizes[0]
        )
        return fundamental_step_size

    def _validate_and_set_step_sizes(self, nodes: List[Node]):
        defined_step_sizes = [
            node.step_size for node in nodes if node.step_size is not None
        ]
        if not defined_step_sizes:
            raise ValueError("At least one node must have a defined step_size")

        fundamental_step_size = self.define_fundamental_step_size(nodes)
        for node in nodes:
            if node.step_size is None:
                node.step_size = fundamental_step_size

        return fundamental_step_size

    def _setup_logger(
        self,
        nodes: List[Node],
        states_to_log: Optional[List[str]],
        logger_cooldown: float,
        fundamental_step_size: float,
    ):
        if not states_to_log:
            self.logger = None
            return

        self.logger = Logger(
            states_to_log, fundamental_step_size, cooldown=logger_cooldown
        )
        nodes.append(self.logger)

    def _get_logger_step_sizes(
        self, nodes: List[Node], states_to_log: List[str]
    ) -> List[float]:
        return [
            node.step_size
            for node in nodes
            if node.step_size is not None
            and any(
                state.search_by_path(path)
                for state in node.state.get_all_states()
                for path in states_to_log
            )
        ]

    def _initialize_graph(self, nodes: List[Node], fundamental_step_size: float):
        self.nodes = nodes + [Clock(fundamental_step_size), StepCounter(nodes)]
        states = reduce(
            lambda x, y: x + y, [node.state.get_all_states() for node in self.nodes]
        )

        # Find nodes that need reset modification
        reset_map = {}
        for node in self.nodes:
            for input_path in node.state.paths:
                if input_path.startswith("reset_"):

                    target_node_name = input_path[6:]  # Remove 'reset_' prefix
                    reset_map[target_node_name] = True

        from regelum.environment.transistor import ResetModifier

        reset_modifier = ResetModifier()

        # Apply reset modifiers where needed
        for node in self.nodes:
            if node.state.name in reset_map:
                if node.transistor is None:
                    node.default_transistor_configuration["transistor"] = (
                        reset_modifier.apply_class(
                            node.default_transistor_configuration["transistor"]
                        )
                    )
        for node in self.nodes:
            node.inputs.resolve(states)

        self.ordered_nodes = self.resolve(self.nodes)
        self._log_node_order()

        for node in self.ordered_nodes:
            if node.transistor is None:
                if not node.is_continuous:
                    node.default_transistor_configuration["transistor"] = (
                        SampleAndHoldModifier(node.step_size).apply_class(
                            node.default_transistor_configuration["transistor"]
                        )
                    )

                node.with_transistor(
                    node.default_transistor_configuration["transistor"],
                    **node.default_transistor_configuration["transistor_kwargs"],
                )

    def _log_node_order(self):
        self.ordered_nodes_str = " -> ".join(
            [node.state.name for node in self.ordered_nodes]
        )
        print(f"Resolved node order: {self.ordered_nodes_str}")

    @staticmethod
    def resolve(nodes: List[Node]) -> List[Node]:
        """Resolves node execution order based on input dependencies."""
        # Create unique identifiers for nodes
        node_map = {}
        for idx, node in enumerate(nodes):
            unique_id = f"{node.state.name}_{idx}"
            node_map[unique_id] = {
                "node": node,
                "state": node.state,
                "inputs": node.inputs.paths_to_states,
                "is_root": node.is_root,
            }

        ordered_ids: List[str] = []
        max_iterations = len(node_map)
        iterations = 0

        def is_input_available(input_path: str, resolved_states: List[State]) -> bool:
            for resolved_state in resolved_states:
                if resolved_state.search_by_path(input_path) is not None:
                    return True
            return False

        # Resolve order
        while len(ordered_ids) < len(node_map):
            if iterations >= max_iterations:
                unresolved = set(node_map.keys()) - set(ordered_ids)
                raise ValueError(
                    f"Circular dependency detected. Unresolved nodes: {unresolved}"
                )

            resolved_states = [node_map[n]["state"] for n in ordered_ids]

            for unique_id, info in node_map.items():
                if unique_id not in ordered_ids:
                    inputs_available = all(
                        is_input_available(input_path, resolved_states)
                        for input_path in info["inputs"]
                    )
                    if inputs_available or info["is_root"]:
                        ordered_ids.append(unique_id)

            iterations += 1

        # Map back to original nodes
        return [node_map[unique_id]["node"] for unique_id in ordered_ids]

    def step(self):
        """Execute a single time step for all nodes in the graph in resolved order."""
        for node in self.ordered_nodes:
            if node.transistor:
                node.transistor.step()
            else:
                raise ValueError(f"Node {node.state.name} does not have a transistor.")

    def reset(self, states_to_reset: Optional[List[str]] = None) -> None:
        """Reset specified states or all nodes if none specified.

        Args:
            states_to_reset: List of state paths to reset. If None, resets all nodes.
        """
        if states_to_reset is None:
            for node in self.nodes:
                node.reset()
            return

        found_states = set()
        for node in self.nodes:
            if reset_states := node.reset(states_to_reset):
                found_states.update(reset_states)

        if missing := set(states_to_reset) - found_states:
            raise ValueError(f"Could not find states: {missing}")


class Clock(Node):
    """Time management node.

    Args:
        nodes: List of nodes to synchronize
        time_start: Initial time value
    """

    state = State("Clock", (1,), np.array([0]))

    def __init__(self, fundamental_step_size: float) -> None:
        """Initialize the Clock node.

        Args:
            fundamental_step_size: Fundamental step size.
        """
        self.fundamental_step_size = fundamental_step_size

        super().__init__(is_root=False, step_size=self.fundamental_step_size)

    def compute_state_dynamics(self) -> Dict[str, RgArray]:
        return {"Clock": self.state.data + self.fundamental_step_size}


class StepCounter(Node):
    """Counts steps in the simulation.

    Args:
        nodes: List of nodes to track
        start_count: Initial counter value
    """

    state = State("step_counter", (1,), np.array([0]))

    def __init__(self, nodes: List[Node], start_count: int = 0) -> None:
        self.state.data = np.array([start_count])
        step_sizes = [node.step_size for node in nodes if not node.is_continuous]
        min_step_size = min(step_sizes)
        super().__init__(is_root=False, step_size=min_step_size)
        from regelum.environment.transistor import Transistor

        self.with_transistor(Transistor)

    def compute_state_dynamics(self) -> Dict[str, RgArray]:
        return {"step_counter": self.state.data + 1}


class Logger(Node):
    """State recording node.

    Args:
        states_to_log: State paths to record
        step_size: Logging interval
        cooldown: Minimum time between logs
    """

    def __init__(
        self, states_to_log: List[str], step_size: float, cooldown: float = 0.0
    ) -> None:
        """Initialize the Logger node.

        Args:
            states_to_log: State paths to record.
            step_size: Logging interval.
            cooldown: Minimum time between logs.
        """
        self.states_to_log = states_to_log
        # Create hierarchical state structure
        self.state = State(
            "Logger",
            None,
            [
                State("counter", (1,), np.array([0.0])),
                State("last_log_time", (1,), np.array([-float("inf")])),
            ],
        )

        self.cooldown = cooldown
        self.inputs = Inputs(["Clock"] + states_to_log)
        super().__init__(is_root=False, step_size=step_size, inputs=self.states_to_log)
        from regelum.environment.transistor import Transistor

        self.with_transistor(Transistor)

        self.logs = {path: [] for path in states_to_log}
        self.logs["time"] = []

        # Setup logging - modified to prevent duplication
        self.logger = logging.getLogger(__name__)
        self.logger.handlers = []  # Clear any existing handlers
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # Prevent propagation to root logger

    def compute_state_dynamics(self) -> Dict[str, Any]:
        inputs = self.inputs.collect()
        current_time = float(inputs["Clock"][0])
        last_log_time = float(self.state["Logger/last_log_time"].data[0])

        if current_time - last_log_time >= self.cooldown:
            self.logs["time"].append(current_time)
            # Log data and build message as before
            log_parts = [f"t={current_time:.3f}"]
            for path in self.states_to_log:
                value = inputs[path]
                self.logs[path].append(value)
                if isinstance(value, np.ndarray):
                    formatted_value = np.array2string(
                        value, precision=3, suppress_small=True
                    )
                else:
                    formatted_value = f"{value:.3f}"
                log_parts.append(f"{path}={formatted_value}")

            self.logger.info(" | ".join(log_parts))
            return {
                "Logger/counter": self.state["Logger/counter"].data + 1,
                "Logger/last_log_time": np.array([current_time]),
            }

        return {
            "Logger/counter": self.state["Logger/counter"].data,
            "Logger/last_log_time": self.state["Logger/last_log_time"].data,
        }


class MPCNodeFactory(Node):
    """Model Predictive Control node.

    Args:
        target_node: Node to control
        control_shape: Control input dimension
        prediction_horizon: MPC horizon length
        is_root: Whether node is a root node
        step_size: Control interval
        input_bounds: Control input constraints
        state_weights: State cost weights
        input_weights: Control cost weights
    """

    def __init__(
        self,
        target_node: Node,
        control_shape: int,
        prediction_horizon: int = 10,
        is_root: bool = False,
        step_size: Optional[float] = None,
        input_bounds: Optional[Dict[str, tuple[float, float]]] = None,
        state_weights: Optional[Dict[str, float]] = None,
        input_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialize the MPC node.

        Args:
            target_node: Node to control.
            control_shape: Control input dimension.
            prediction_horizon: MPC horizon length.
            is_root: Whether node is a root node.
            step_size: Control interval.
            input_bounds: Control input constraints.
            state_weights: State cost weights.
            input_weights: Control cost weights.
        """
        if step_size is None:
            step_size = target_node.step_size

        assert hasattr(
            target_node, "system_dynamics"
        ), "Target node must have a system dynamics method of the form system_dynamics(x, u) -> Dict[str, RgArray]"

        self.control_shape = control_shape
        self.target_node = target_node
        self.state = State(
            f"mpc_{target_node.state.name}_control", (self.control_shape,)
        )
        self.inputs = Inputs([target_node.state.name, "Clock"])

        super().__init__(is_root, step_size)

        self.N = prediction_horizon
        self.dt = self.step_size
        self.state_weights = state_weights or {target_node.state.name: 1.0}
        self.input_weights = input_weights or {self.state.name: 0.01}
        self.input_bounds = input_bounds

    def with_transistor(self, transistor: Type[Transistor], **transistor_kwargs):
        self.setup_optimization(self.input_bounds or {})
        return super().with_transistor(transistor, **transistor_kwargs)

    def setup_optimization(self, input_bounds: Dict[str, tuple[float, float]]) -> None:
        import regelum as rg

        with rg.symbolic_inference():
            state_dynamics = self.target_node.compute_state_dynamics()
            state_dim = state_dynamics[self.target_node.state.name].shape[0]
            input_dim = self.state.data.shape[0]

        self.opti = cs.Opti()
        self.X = self.opti.variable(state_dim, self.N + 1)
        self.U = self.opti.variable(input_dim, self.N)
        self.x0 = self.opti.parameter(state_dim)

        objective = 0
        for k in range(self.N):
            state_error = self.X[:, k]
            objective += sum(
                w * (state_error[i]) ** 2
                for i in range(state_dim)
                for _, w in self.state_weights.items()
            )
            objective += sum(
                w * (self.U[:, k][i]) ** 2
                for i in range(input_dim)
                for _, w in self.input_weights.items()
            )

        self.opti.minimize(objective)

        # Initial condition
        self.opti.subject_to(self.X[:, 0] == self.x0)

        # System dynamics
        for k in range(self.N):
            dynamics = self.target_node.system_dynamics(self.X[:, k], self.U[:, k])
            x_next = self.X[:, k] + self.dt * dynamics[self.target_node.state.name]
            self.opti.subject_to(self.X[:, k + 1] == x_next)

        # Input bounds
        if input_bounds:
            lb, ub = input_bounds.get(self.state.name, (-float("inf"), float("inf")))
            self.opti.subject_to(self.opti.bounded(lb, self.U, ub))

        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}
        self.opti.solver("ipopt", opts)

    def compute_state_dynamics(self):
        current_state = self.inputs[self.target_node.state.name].data
        self.opti.set_value(self.x0, current_state)

        sol = self.opti.solve()
        u_optimal = sol.value(self.U[:, 0])

        return {self.state.name: u_optimal}
