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
    TYPE_CHECKING,
    Union,
    List,
    Any,
    Type,
    TypeVar,
    overload,
    Literal,
    cast,
)

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

if TYPE_CHECKING:
    from .transistor import Transistor
from regelum.typing import (
    RgArray,
)
from math import gcd

T = TypeVar("T")


@dataclass
class State:
    """Hierarchical state container with path-based access.

    Args:
        name: State identifier
        shape: Shape of the state data
        _value: State data or list of child states
        is_leaf: Auto-determined leaf status
    """

    name: str
    shape: Optional[Tuple[int, ...]] = None
    _value: Union[Any, List["State"], None] = None
    is_leaf: bool = field(init=False)

    def __post_init__(self):
        self.is_leaf = self._determine_leaf_status()
        self._validate_hierarchical_state()
        self._path_cache = {}
        self._build_path_cache()

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

    def get_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Get the shapes of all leaf states."""
        return {path: self[path].shape for path in self.paths}

    def _build_path_cache(self):
        """Build a cache of all possible paths to substates"""

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
            missing = set(self.paths_to_states) - {state.name for state in found_states}
            raise ValueError(f"Could not resolve all input paths. Missing: {missing}")

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
            return {state.name: state.data for state in self.states}
        else:
            return {}

    def __getitem__(self, key: str) -> State:
        """Get a resolved input state by its name."""
        if not self._resolved:
            raise ValueError("Resolve inputs before accessing them")
        try:
            index = self.paths_to_states.index(key)
            return self.states[index]
        except ValueError:
            raise KeyError(f"Input '{key}' not found in paths_to_states")


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
    ) -> None:
        """Instantiate a Node object."""
        if not hasattr(self, "state"):
            if state is None:
                raise ValueError("State must be fully specified.")
            self.state = state

        if not hasattr(self, "inputs"):
            if inputs is None:
                inputs = []
            self.inputs = Inputs(inputs)
        else:
            if isinstance(self.inputs, list):
                self.inputs = Inputs(self.inputs)
            elif isinstance(self.inputs, Inputs):
                pass
            else:
                raise ValueError("Inputs must be a list of strings")

        if self.state is None:
            raise ValueError("State must be fully specified.")

        self.is_root = is_root
        self.step_size = step_size
        if self.is_root:
            assert (
                self.state.is_defined
            ), f"Initial state must be defined for the root node {self.state.name}"
        self.transistor = None

    def with_transistor(self, transistor: Type[Transistor], **transistor_kwargs):
        self.transistor = transistor(node=self, **transistor_kwargs)
        return self

    @abstractmethod
    def compute_state_dynamics(self) -> Dict[str, Any]:
        """Compute the state dynamics given inputs."""
        pass


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
        self._validate_and_set_step_sizes(nodes)
        self._setup_logger(nodes, states_to_log, logger_cooldown)
        self._initialize_graph(nodes)

    def _validate_and_set_step_sizes(self, nodes: List[Node]):
        defined_step_sizes = [
            node.step_size for node in nodes if node.step_size is not None
        ]
        if not defined_step_sizes:
            raise ValueError("At least one node must have a defined step_size")

        min_step_size = min(defined_step_sizes)
        for node in nodes:
            if node.step_size is None:
                node.step_size = min_step_size

    def _setup_logger(
        self,
        nodes: List[Node],
        states_to_log: Optional[List[str]],
        logger_cooldown: float,
    ):
        if not states_to_log:
            self.logger = None
            return

        step_sizes = self._get_logger_step_sizes(nodes, states_to_log)
        min_step_size = min(step_sizes) if step_sizes else nodes[0].step_size
        self.logger = Logger(states_to_log, min_step_size, cooldown=logger_cooldown)
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

    def _initialize_graph(self, nodes: List[Node]):
        self.nodes = nodes + [Clock(nodes)]
        states = reduce(
            lambda x, y: x + y, [node.state.get_all_states() for node in self.nodes]
        )

        for node in self.nodes:
            node.inputs.resolve(states)

        self.ordered_nodes = self.resolve(self.nodes)
        self._log_node_order()

    def _log_node_order(self):
        self.ordered_nodes_str = " -> ".join(
            [node.state.name for node in self.ordered_nodes]
        )
        print(f"Resolved node order: {self.ordered_nodes_str}")

    @staticmethod
    def resolve(nodes: List[Node]) -> List[Node]:
        """Resolves node execution order based on input dependencies."""
        # Create mapping of node info
        node_map = {
            node.state.name: {
                "state": node.state,
                "inputs": node.inputs.states,
                "is_root": node.is_root,
            }
            for node in nodes
        }

        # Check for duplicates
        assert len(set(node_map)) == len(node_map), "Duplicate node states detected"

        ordered_names: List[str] = []
        max_iterations = len(node_map)
        iterations = 0

        # Resolve order
        while len(ordered_names) < len(node_map):
            if iterations >= max_iterations:
                unresolved = node_map.keys() - set(ordered_names)
                raise ValueError(
                    f"Circular dependency detected. Unresolved nodes: {unresolved}"
                )

            resolved_states = [node_map[n]["state"] for n in ordered_names]

            for name, info in node_map.items():
                if name not in ordered_names:
                    inputs_available = all(
                        input_state in resolved_states for input_state in info["inputs"]
                    )
                    if inputs_available or info["is_root"]:
                        ordered_names.append(name)

            iterations += 1

        # Map back to original nodes
        return [
            next(n for n in nodes if n.state.name == name) for name in ordered_names
        ]

    def step(self):
        """Execute a single time step for all nodes in the graph in resolved order."""
        for node in self.ordered_nodes:
            if node.transistor:
                node.transistor.step()
            else:
                raise ValueError(f"Node {node.state.name} does not have a transistor.")


class Clock(Node):
    """Time management node.

    Args:
        nodes: List of nodes to synchronize
        time_start: Initial time value
    """

    state = State("Clock", (1,))

    def __init__(self, nodes: List[Node], time_start: float = 0.0) -> None:
        """Instantiate a Clock node with a fixed time step size."""
        step_sizes = [node.step_size for node in nodes]

        def float_gcd(a: float, b: float) -> float:
            precision = 1e-9
            a, b = round(a / precision), round(b / precision)
            return gcd(int(a), int(b)) * precision

        self.fundamental_step_size = (
            reduce(float_gcd, step_sizes) if len(set(step_sizes)) > 1 else step_sizes[0]
        )

        self.state.data = np.array([time_start])
        super().__init__(is_root=False, step_size=self.fundamental_step_size)
        from regelum.environment.transistor import Transistor

        self.with_transistor(Transistor)

    def compute_state_dynamics(self) -> Dict[str, RgArray]:
        return {"Clock": self.state.data + self.fundamental_step_size}


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
        self.states_to_log = states_to_log
        self.state = State("Logger", (1,))
        self.state.data = np.array([0.0])
        self.cooldown = cooldown
        self.last_log_time = -float("inf")  # Ensure first log happens immediately

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

        if current_time - self.last_log_time >= self.cooldown:
            self.logs["time"].append(current_time)
            self.last_log_time = current_time

            # Build log message
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

        return {"Logger": self.state.data}


class MPCNode(Node):
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

        try:
            sol = self.opti.solve()
            u_optimal = sol.value(self.U[:, 0])
        except:
            u_optimal = np.zeros(self.control_shape)

        return {self.state.name: u_optimal}
