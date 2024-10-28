"""This module contains the base class for all nodes in the environment."""

from __future__ import annotations
from functools import reduce
from typing import Dict, Optional, Tuple, TYPE_CHECKING, Union, List, Any, Type
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
import casadi as cs
import numpy as np
from regelum import _SYMBOLIC_INFERENCE_ACTIVE

if TYPE_CHECKING:
    from .transistor import Transistor
from regelum.typing import (
    RgArray,
)
from math import gcd


@dataclass
class State:
    """A wrapper class for the state of a node in the environment."""

    name: str
    shape: Optional[Tuple[int, ...]] = None
    _value: Optional[Union[Any, List["State"]]] = (
        None  # Use _value to store the actual value
    )
    is_leaf: bool = field(init=False)

    def __post_init__(self):
        if (
            isinstance(self._value, list)
            and len(self._value) > 0
            and all(isinstance(s, State) for s in self._value)
        ):
            self.is_leaf = False
        else:
            self.is_leaf = True
        # If _value is a list, but not all elements are State instances, and it's supposed to be hierarchical
        if not self.is_leaf and not all(isinstance(s, State) for s in self._value):
            raise TypeError(
                f"The _value of a hierarchical State '{self.name}' must be a list of State instances."
            )

    @property
    def value(self):
        """Return a dict representation of the state."""
        symbolic = getattr(_SYMBOLIC_INFERENCE_ACTIVE, "value", False)

        if self.is_leaf:
            # Leaf state
            val = self.to_casadi_symbolic() if symbolic else self._value
            return {
                "name": self.name,
                "shape": self.shape,
                "value": val,
            }
        else:
            # Hierarchical state
            return {
                "name": self.name,
                "shape": self.shape,
                "states": [substate.value for substate in self._value],
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
        """Search for a substate by its path."""
        path_parts = [part for part in path.split("/") if part]
        if not path_parts:
            return None

        current_node = path_parts[0]
        remaining_path = "/".join(path_parts[1:])

        if current_node != self.name:
            return None

        if not remaining_path:
            return self

        if self.is_leaf:
            return None

        for substate in self._value:
            if result := substate.search_by_path(remaining_path):
                return result

        return None

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
            for substate in self._value:
                substate._collect_paths(prefix=current_path, paths=paths)

    def get_all_states(self) -> List["State"]:
        """Get a list of all leaf states."""
        states = []
        self._collect_states(states=states)
        return states

    def _collect_states(self, states: List["State"]):
        """Helper method to collect states recursively."""
        if self.is_leaf:
            states.append(self)
        else:
            for substate in self._value:
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


@dataclass
class Inputs:
    """A wrapper class for the inputs of a node in the environment."""

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
            return {state.name: state.value["value"] for state in self.states}
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
    """An entity representing an atomic unit with time-dependent state."""

    inputs: Optional[Union[List[str], Inputs]] = None
    state: Optional[State] = None
    is_root: bool = False

    def __init__(self, is_root: bool = False) -> None:
        """Instantiate a Node object."""
        if self.inputs is not None:
            self.inputs = (
                Inputs(self.inputs) if isinstance(self.inputs, list) else self.inputs
            )
        else:
            self.inputs = Inputs([])

        if self.state is None:
            raise ValueError("State must be fully specified.")

        self.is_root = is_root
        if self.is_root:
            assert (
                self.state.is_defined
            ), f"Initial state must be defined for the root node {self.state.name}"
        self.transistor = None  # Transistor will be set later

    def with_transistor(self, transistor: Type[Transistor], **transistor_kwargs):
        self.transistor = transistor(node=self, **transistor_kwargs)
        return self

    @abstractmethod
    def compute_state_dynamics(
        self, state: Dict[str, Any], inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute the state dynamics given inputs."""
        pass


class Graph:
    def __init__(self, nodes: List[Node]) -> None:
        self.nodes = nodes
        # Collect all states from nodes
        states: List[State] = reduce(
            lambda x, y: x + y, [node.state.get_all_states() for node in nodes]
        )
        # Resolve inputs for each node
        for node in self.nodes:
            node.inputs.resolve(states)

        self.ordered_nodes = self.resolve(self.nodes)
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
    """A node representing a clock with a fixed time step size."""

    state = State("Clock", (1,))

    def __init__(self, nodes: List[Node], time_start: float = 0.0) -> None:
        """Instantiate a Clock node with a fixed time step size."""
        step_sizes = [node.transistor.step_size for node in nodes]

        def float_gcd(a: float, b: float) -> float:
            precision = 1e-9
            a, b = round(a / precision), round(b / precision)
            return gcd(int(a), int(b)) * precision

        self.fundamental_step_size = (
            reduce(float_gcd, step_sizes) if len(set(step_sizes)) > 1 else step_sizes[0]
        )

        self.state.value = np.array([time_start])
        super().__init__(state=self.state)
        self.with_transistor(Transistor, step_size=self.fundamental_step_size)

    def compute_state_dynamics(self, inputs: Dict[str, RgArray]) -> Dict[str, RgArray]:
        assert isinstance(self.state.value, np.ndarray)
        return {"Clock": self.state.value[0] + self.fundamental_step_size}


class Terminate(Node):
    """A node representing a termination condition."""

    postfix = "_terminate"
    inputs = Inputs(["Clock", "plant"])

    def __init__(self, node_to_terminate: Node) -> None:
        """Instantiate a Terminate node."""
        self.node_to_terminate = node_to_terminate
        self.state = State(node_to_terminate.state.name + self.postfix, (1,))
        super().__init__(state=self.state, inputs=self.inputs)
        self.with_transistor(Transistor, step_size=0.01)

    def compute_state_dynamics(self, inputs: Dict[str, RgArray]) -> Dict[str, bool]:
        if self.node_to_terminate.transistor.time_final is not None:
            return {
                self.state.name: (
                    False
                    if inputs["Clock"] < self.node_to_terminate.transistor.time_final
                    else True
                )
            }
        return {self.state.name: False}
