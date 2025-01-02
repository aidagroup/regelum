"""Base classes for node-based computation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from functools import reduce
from math import gcd
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypedDict,
)

import casadi as cs
import numpy as np
import torch
from functools import wraps

from regelum.utils import find_scc
from regelum.utils.logger import logger

if TYPE_CHECKING:
    from regelum.environment.node.parallel import ParallelGraph

FullName = str


class ResolveStatus(StrEnum):
    """Status of the resolve operation."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNDEFINED = "undefined"


class Metadata(TypedDict):
    """Metadata for a variable."""

    initial_value: Any
    symbolic_value: Optional[cs.MX]
    shape: Optional[Tuple[int, ...]]
    reset_modifier: Optional[Callable[..., Any]]


def default_metadata() -> Metadata:
    return {
        "initial_value": None,
        "symbolic_value": None,
        "shape": None,
        "reset_modifier": None,
    }


@dataclass(slots=True)
class Variable:
    """A variable with optional metadata."""

    name: str
    value: Optional[Any] = None
    metadata: Metadata = field(default_factory=default_metadata)
    node_name: str = field(default="")

    def __post_init__(self) -> None:
        if (
            "initial_value" not in self.metadata
            or self.metadata["initial_value"] is None
        ):
            self.metadata["initial_value"] = deepcopy(self.value)

    def __deepcopy__(self, memo: Dict) -> Variable:
        return Variable(
            name=self.name,
            value=deepcopy(self.value, memo),
            metadata=deepcopy(self.metadata, memo),
            node_name=self.node_name,
        )

    def reset(self, apply_reset_modifier: bool = True) -> None:
        if (
            apply_reset_modifier
            and "reset_modifier" in self.metadata
            and self.metadata["reset_modifier"]
        ):
            self.value = deepcopy(
                self.metadata["reset_modifier"](self.metadata["initial_value"])
            )
        else:
            self.value = deepcopy(self.metadata["initial_value"])

    def to_casadi_symbolic(self) -> Optional[cs.MX]:
        """Create or return symbolic representation."""
        if self.metadata["symbolic_value"] is None:
            shape = self._infer_shape()
            if shape:
                self.metadata["symbolic_value"] = cs.MX.sym(str(self.name), *shape)  # type: ignore[arg-type]
        return self.metadata["symbolic_value"]

    def _infer_shape(self) -> Optional[Tuple[int, ...]]:
        """Infer shape from value or metadata."""
        if self.metadata["shape"]:
            return self.metadata["shape"]

        if hasattr(self.value, "shape"):
            if isinstance(self.value, (np.ndarray, torch.Tensor, cs.DM)):
                return self.value.shape

        if isinstance(self.value, (int, float, bool)):
            return (1,)

        return None

    def get_value(self) -> Any:
        """Return symbolic or actual value."""
        from regelum import _SYMBOLIC_INFERENCE_ACTIVE

        return (
            self.to_casadi_symbolic()
            if getattr(_SYMBOLIC_INFERENCE_ACTIVE, "value", False)
            else self.value
        )

    @property
    def full_name(self) -> str:
        """Return fully qualified name."""
        return f"{self.node_name}.{self.name}"

    def set_new_value(self, value: Any) -> None:
        """Set new value and update initial value."""
        self.value = value
        self.metadata["initial_value"] = deepcopy(value)


@dataclass(slots=True, frozen=True)
class Inputs:
    """Collection of input variable names."""

    inputs: List[str]  # List of full names (node_name.var_name)

    def resolve(
        self, variables: List[Variable]
    ) -> Tuple[ResolvedInputs, Set[FullName]]:
        """Map input names to actual variables."""
        var_dict = {var.full_name: var for var in variables}
        resolved = []
        unresolved = set()

        for name in self.inputs:
            if name in var_dict:
                resolved.append(var_dict[name])
            else:
                unresolved.add(name)

        return ResolvedInputs(inputs=resolved), unresolved


@dataclass(slots=True, frozen=True)
class ResolvedInputs:
    """Collection of resolved input variables."""

    inputs: List[Variable]

    def find(self, full_name: str) -> Optional[Variable]:
        """Find variable by full name."""
        node_name, var_name = full_name.split(".")

        for var in self.inputs:
            if var.full_name == full_name:
                return var

            if (var.name in var_name or var_name in var.name) and (
                var.node_name in node_name or node_name in var.node_name
            ):
                return var

        return None

    def __len__(self) -> int:
        return len(self.inputs)


class Node(ABC):
    """Base unit of computation."""

    _instances: Dict[str, List[Node]] = {}

    def __new__(cls, *args: Any, **kwargs: Any) -> Node:
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
    ):
        """Initialize node with inputs and configuration."""
        self.inputs = self._normalize_inputs(inputs)
        self.step_size = step_size
        self.is_continuous = is_continuous
        self.last_update_time = None
        self.is_root = is_root
        self._variables: List[Variable] = []
        self.resolved_inputs: Optional[ResolvedInputs] = None

        if is_continuous and not hasattr(self, "state_transition_map"):
            raise ValueError(
                f"Continuous node {self.__class__.__name__} must implement state_transition_map"
            )

        self._internal_name = name or self.__class__.__name__.lower()
        self._external_name = f"{self._internal_name}_{self.get_instance_count()}"

    def _normalize_inputs(self, inputs: Optional[Inputs | List[str]]) -> Inputs:
        """Convert inputs to Inputs instance."""
        if not hasattr(self, "inputs"):
            if inputs is None:
                return Inputs([])
            return Inputs(inputs) if isinstance(inputs, list) else inputs
        return Inputs(self.inputs) if isinstance(self.inputs, list) else self.inputs

    @classmethod
    def get_instances(cls) -> List[Node]:
        """Get all instances of this class."""
        return cls._instances.get(cls.__name__, [])

    @classmethod
    def get_instance_count(cls) -> int:
        """Get count of instances for this class."""
        return len(cls._instances.get(cls.__name__, []))

    @property
    def name(self) -> str:
        """Internal name for self-reference."""
        return self._internal_name

    @property
    def external_name(self) -> str:
        """External name for unique identification."""
        return self._external_name

    @property
    def variables(self) -> List[Variable]:
        """List of node variables."""
        return self._variables

    @property
    def is_resolved(self) -> bool:
        """Whether inputs are resolved to variables."""
        return self.resolved_inputs is not None

    def define_variable(
        self,
        name: str,
        value: Optional[Any] = None,
        metadata: Optional[Metadata] = None,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> Variable:
        """Create and register a variable."""
        meta = metadata or default_metadata()
        if shape is not None:
            meta["shape"] = shape
        var = Variable(name, value, meta, self.external_name)
        self._variables.append(var)
        return var

    def find_variable(self, name: str) -> Optional[Variable]:
        """Find variable by name."""
        return next((var for var in self._variables if var.name == name), None)

    def get_variable(self, name: str) -> Variable:
        """Get variable by name or raise error."""
        if var := self.find_variable(name):
            return var
        raise ValueError(f"Variable '{name}' not found in node '{self.external_name}'")

    def reset(
        self,
        variables_to_reset: Optional[List[str]] = None,
        apply_reset_modifier: bool = True,
    ) -> None:
        """Reset specified or all variables."""
        if variables_to_reset is None:
            for var in self._variables:
                var.reset(apply_reset_modifier)
            return

        not_found = set(variables_to_reset)
        for var in self._variables:
            if var.name in not_found:
                var.reset(apply_reset_modifier)
                not_found.remove(var.name)

        if not_found:
            raise ValueError(
                f"Variables {not_found} not found in node '{self.external_name}'"
            )

    def get_full_names(self) -> List[str]:
        """Get fully qualified names of all variables."""
        return [f"{self.external_name}.{var.name}" for var in self._variables]

    def get_resolved_inputs(
        self, variables: List[Variable]
    ) -> Tuple[ResolvedInputs, Set[FullName]]:
        """Resolve inputs to variables."""
        return self.inputs.resolve(variables)

    def alter_input_names(self, mapping: Dict[str, str]) -> None:
        """Update input names using mapping."""
        self.inputs = Inputs([mapping.get(name, name) for name in self.inputs.inputs])

    def alter_variable_names(self, mapping: Dict[str, str]) -> None:
        """Update variable names using mapping."""
        for var in self._variables:
            var.name = mapping.get(var.name, var.name)

    def resolve(self, variables: List[Variable]) -> ResolvedInputs:
        """Resolve node inputs."""
        self.resolved_inputs, self.unresolved_inputs = self.get_resolved_inputs(
            variables
        )
        if self.unresolved_inputs:
            raise ValueError(
                f"Couldn't resolve inputs {self.unresolved_inputs} for node {self.external_name}"
            )
        return self.resolved_inputs

    def alter_name(self, new_name: str) -> str:
        """Update node name and propagate to variables."""
        old_name = self.external_name
        self._internal_name = new_name
        self._external_name = f"{new_name}_{self.get_instance_count()}"
        for var in self._variables:
            var.node_name = self.external_name
        return old_name

    def __or__(self, other: Node) -> Node:
        """Connect nodes using | operator."""
        nodes = []
        if isinstance(self, Graph):
            nodes.extend(self.nodes)
        else:
            nodes.append(self)

        if isinstance(other, Graph):
            nodes.extend(other.nodes)
        else:
            nodes.append(other)

        return Graph(nodes)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.external_name}, inputs={self.inputs}, variables={self.variables})"

    def __repr__(self) -> str:
        return self.__str__()

    @abstractmethod
    def step(self) -> None:
        """Perform computation step."""
        pass

    def __deepcopy__(self, memo: Dict) -> Node:
        """Custom deepcopy implementation to handle instance counting."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k == "_variables":
                vars_copy = [deepcopy(var, memo) for var in v]
                setattr(result, k, vars_copy)
            elif k == "resolved_inputs":
                setattr(result, k, None)
            else:
                setattr(result, k, deepcopy(v, memo))

        if id(result) not in [id(x) for x in cls._instances.get(cls.__name__, [])]:
            cls._instances.setdefault(cls.__name__, []).append(result)
        result._external_name = (
            f"{result._internal_name}_{len(cls._instances[cls.__name__])}"
        )
        if not isinstance(result, Graph):
            for var in result._variables:
                var.node_name = result._external_name

        return result


class Semaphore(Node):
    def __init__(self, name: str = "semaphore", inputs: Optional[Inputs] = None):
        super().__init__(inputs=inputs, name=name)
        self.flag = self.define_variable("flag", value=False)

    def step(self):
        raise NotImplementedError("Semaphore step not implemented")


class Reset(Node):
    def __init__(self, name: str = "reset", inputs: Optional[Inputs] = None):
        super().__init__(inputs=inputs, name=name)
        self.flag = self.define_variable("flag", value=False)

    def step(self):
        raise NotImplementedError("ResetSemaphore step not implemented")


class Graph(Node):
    """A graph of connected nodes."""

    def __init__(
        self,
        nodes: List[Node],
        debug: bool = False,
        n_step_repeats: int = 1,
        initialize_inner_time: bool = False,
        states_to_log: Optional[List[str]] = None,
        logger_cooldown: float = 0.0,
        name: str = "graph",
    ) -> None:
        """Initialize Graph node."""
        super().__init__(name=name)
        fundamental_step_size = self._validate_and_set_step_sizes(nodes)
        self.step_size = fundamental_step_size
        if initialize_inner_time:
            from regelum.environment.node.library.logging import Clock, StepCounter

            self.clock = clock = Clock(fundamental_step_size)
            step_counter = StepCounter([clock], start_count=0)
            nodes.append(step_counter)
            nodes.append(clock)
            self._align_discrete_nodes_execution_with_step_size(
                [
                    discrete_node
                    for discrete_node in nodes
                    if not discrete_node.is_continuous
                    and not isinstance(discrete_node, Clock)
                ]
            )
        self._process_resets(nodes)
        if states_to_log:
            self._setup_logger(
                nodes, states_to_log, logger_cooldown, fundamental_step_size
            )
        self.nodes = nodes
        self.debug = debug
        self.n_step_repeats = n_step_repeats
        self._collect_node_data()
        self.resolve_status = ResolveStatus.UNDEFINED

    def _process_resets(self, nodes: List[Node]):
        reset_semaphores = [node for node in nodes if isinstance(node, Reset)]
        for reset_semaphore in reset_semaphores:
            corresponding_node = next(
                (
                    node
                    for node in nodes
                    if node.external_name
                    == "_".join(reset_semaphore.name.split("_")[1:])
                ),
                None,
            )
            if corresponding_node:
                self.apply_reset_modifier(corresponding_node, reset_semaphore)

    def apply_reset_modifier(self, node: Node, reset_semaphore: Node):
        ResetModifier(node, reset_semaphore).bind_to_node(node)

    def _align_discrete_nodes_execution_with_step_size(
        self, discrete_nodes: List[Node]
    ) -> None:
        if not hasattr(self, "clock"):
            raise ValueError("Clock not found in graph")

        for node in discrete_nodes:
            ZeroOrderHold(node, self.clock).bind_to_node(node)

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
        from regelum.environment.node.library.logging import Logger

        self.logger = Logger(
            states_to_log, fundamental_step_size, cooldown=logger_cooldown
        )
        nodes.append(self.logger)

    def define_fundamental_step_size(self, nodes: List[Node]):
        step_sizes = [node.step_size for node in nodes if node.step_size is not None]

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

    def _collect_node_data(self) -> None:
        """Collect inputs and variables from all nodes."""
        # Collect external inputs (not provided by any node in the graph)
        provided_vars = {
            f"{node.external_name}.{var.name}"
            for node in self.nodes
            for var in node.variables
        }
        external_inputs = []

        for node in self.nodes:
            if isinstance(node.inputs, Inputs):
                external_inputs.extend(
                    name for name in node.inputs.inputs if name not in provided_vars
                )

        self.inputs = Inputs(list(set(external_inputs)))

        if any(node.resolved_inputs for node in self.nodes):
            resolved_vars = [
                var
                for node in self.nodes
                if node.resolved_inputs
                for var in node.resolved_inputs.inputs
            ]
            self.resolved_inputs = ResolvedInputs(resolved_vars)

        self._variables = [var for node in self.nodes for var in node.variables]

    def parallelize(
        self,
        dashboard_address: str = "0.0.0.0:8787",
        memory_limit: str = "2GB",
        n_workers: Optional[int] = None,
        protocol: str = "tcp://",
        scheduler_sync_interval: int = 2,
        scheduler_port: int = 0,
        silence_logs: int = 30,
        processes: Optional[bool] = True,
        threads_per_worker: int = 1,
    ) -> ParallelGraph:
        """Convert to parallel execution mode."""
        from regelum.environment.node.parallel import ParallelGraph

        return ParallelGraph(
            self.nodes,
            self.debug,
            dashboard_address=dashboard_address,
            memory_limit=memory_limit,
            n_workers=n_workers,
            protocol=protocol,
            scheduler_sync_interval=scheduler_sync_interval,
            scheduler_port=scheduler_port,
            silence_logs=silence_logs,
            processes=processes,
            threads_per_worker=threads_per_worker,
        )  # type: ignore[return-value]

    def step(self) -> None:
        """Execute all nodes in sequence."""
        for _ in range(self.n_step_repeats):
            for node in self.nodes:
                node.step()

    def resolve(self, variables: List[Variable]) -> ResolveStatus:
        """Resolve inputs for all nodes and determine execution order."""
        var_names = {}
        for node in self.nodes:
            for var in node.variables:
                full_name = f"{node.external_name}.{var.name}"
                if full_name in var_names:
                    self.resolve_status = ResolveStatus.FAILURE
                    raise ValueError(
                        f"Duplicate variable name found: {full_name}\n"
                        f"First defined in: {var_names[full_name].__class__.__name__}\n"
                        f"Duplicated in: {node.__class__.__name__}"
                    )
                var_names[full_name] = node

        dependencies: Dict[str, Set[str]] = {}
        providers = {
            f"{node.external_name}.{var.name}": node
            for node in self.nodes
            for var in node.variables
        }

        for node in self.nodes:
            if isinstance(node.inputs, Inputs):
                deps = set()
                for input_name in node.inputs.inputs:
                    if provider := providers.get(input_name):
                        deps.add(provider.external_name)
                dependencies[node.external_name] = deps

        ordered_nodes = self._sort_nodes_by_dependencies(dependencies)
        self.nodes = ordered_nodes

        if self.debug:
            logger.info(f"\nResolved node execution order:\n{self}")

        unresolved = {}
        for node in self.nodes:
            try:
                node.resolve(variables)
            except ValueError as e:
                if "Couldn't resolve inputs" in str(e):
                    unresolved[node.external_name] = set(
                        str(e)
                        .split("{")[1]
                        .split("}")[0]
                        .strip()
                        .replace("'", "")
                        .split(", ")
                    )

        if unresolved:
            self.resolve_status = ResolveStatus.PARTIAL
            if self.debug:
                self._log_unresolved_inputs(unresolved)
        else:
            self.resolve_status = ResolveStatus.SUCCESS

        return self.resolve_status

    def _sort_nodes_by_dependencies(
        self, dependencies: Dict[str, Set[str]]
    ) -> List[Node]:
        """Sort nodes based on dependencies and root status."""
        root_nodes = [node for node in self.nodes if node.is_root]
        non_root_nodes = [node for node in self.nodes if not node.is_root]
        ordered = []
        visited = set()

        def visit(node: Node) -> None:
            if node.external_name in visited:
                return
            visited.add(node.external_name)

            for dep_name in dependencies.get(node.external_name, set()):
                dep_node = next(n for n in self.nodes if n.external_name == dep_name)
                if not dep_node.is_root:
                    visit(dep_node)

            ordered.append(node)

        ordered.extend(root_nodes)
        visited.update(node.external_name for node in root_nodes)

        for node in non_root_nodes:
            visit(node)

        return ordered

    def _log_unresolved_inputs(self, unresolved: Dict[str, Set[str]]) -> None:
        """Log information about unresolved inputs."""
        msg = "\nUnresolved inputs found:"
        for node_name, inputs in unresolved.items():
            msg += f"\n{node_name}:"
            for input_name in inputs:
                msg += f"\n  - {input_name}"
        msg += f"\nStatus: {self.resolve_status}"
        logger.info(msg)

    def insert_node(self, node: Node) -> ResolveStatus:
        """Insert a node into the graph and resolve it."""
        self.nodes.append(node)
        self._collect_node_data()
        if isinstance(node, Graph):
            node._collect_node_data()
            node.resolve(node.variables + self.variables)
        return self.resolve(self.variables)

    def clone_node(self, node_name: str, new_name: Optional[str] = None) -> Node:
        """Clone a node and its variables."""
        original = next(
            (node for node in self.nodes if node.external_name == node_name),
            None,
        )
        if not original:
            available = "\n- ".join(node.external_name for node in self.nodes)
            raise ValueError(
                f"Node '{node_name}' not found in graph.\nAvailable nodes:\n- {available}"
            )

        memo: Dict[int, Any] = {"parent_graph": self}  # type: ignore[assignment]
        cloned = deepcopy(original, memo)

        if new_name:
            cloned.alter_name(new_name)

        self.insert_node(cloned)
        return cloned

    def _update_graph_node_names(self, graph: Graph) -> None:
        """Update names of nodes in a cloned graph."""
        graph_num = int(graph.external_name.split("_")[-1])
        node_mapping = {}
        cloned_bases = set()

        for node in graph.nodes:
            old_name = node.external_name
            base_name = "_".join(old_name.split("_")[:-1])
            new_name = f"{base_name}_{graph_num}"
            node_mapping[old_name] = new_name
            cloned_bases.add(base_name)
            node.alter_name(base_name)

        for node in graph.nodes:
            if isinstance(node.inputs, Inputs):
                new_inputs = []
                for input_name in node.inputs.inputs:
                    provider_name, var_name = input_name.split(".")
                    provider_base = "_".join(provider_name.split("_")[:-1])

                    if provider_name in node_mapping:
                        new_name = node_mapping[provider_name]
                        new_inputs.append(f"{new_name}.{var_name}")
                    elif provider_base in cloned_bases:
                        new_inputs.append(f"{provider_base}_{graph_num}.{var_name}")
                    else:
                        new_inputs.append(input_name)

                node.inputs = Inputs(new_inputs)
                node.resolved_inputs = None

        graph.resolve(graph.variables)
        graph._collect_node_data()

        for node in graph.nodes:
            if not node.is_resolved:
                node.resolve(graph.variables)

    def extract_path_as_graph(self, path: str, n_step_repeats: int = 1) -> Graph:
        """Extract a minimal subgraph containing specified nodes."""
        if not path or "->" not in path:
            raise ValueError("Path must be in format: 'node1 -> node2 -> node3'")

        node_names = [name.strip() for name in path.split("->")]
        if not all(node_names):
            raise ValueError("Empty node names are not allowed")

        # Verify nodes exist
        name_to_node = {node.external_name: node for node in self.nodes}
        missing = [name for name in node_names if name not in name_to_node]
        if missing:
            available = "\n- ".join(sorted(name_to_node.keys()))
            raise ValueError(
                f"Could not find nodes: {missing}\nAvailable nodes:\n- {available}"
            )

        # Build dependency graph
        dependencies = self._build_dependency_graph()
        required_nodes = self._find_required_nodes(node_names, dependencies)
        subgraph_nodes = [
            node for node in self.nodes if node.external_name in required_nodes
        ]

        return Graph(subgraph_nodes, debug=self.debug, n_step_repeats=n_step_repeats)  # type: ignore[return-value]

    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """Build bidirectional dependency graph."""
        dependencies = {node.external_name: set() for node in self.nodes}

        for node in self.nodes:
            if isinstance(node.inputs, Inputs):
                for input_name in node.inputs.inputs:
                    for provider in self.nodes:
                        if any(
                            input_name == f"{provider.external_name}.{var.name}"
                            for var in provider.variables
                        ):
                            dependencies[node.external_name].add(provider.external_name)
                            dependencies[provider.external_name].add(node.external_name)

        return dependencies

    def _find_required_nodes(
        self, path_nodes: List[str], dependencies: Dict[str, Set[str]]
    ) -> Set[str]:
        """Find all nodes required for the path."""
        required = set()

        for i in range(len(path_nodes) - 1):
            start, end = path_nodes[i], path_nodes[i + 1]
            if end not in dependencies.get(start, set()):
                path = self._find_path(start, end, dependencies)
                if not path:
                    raise ValueError(f"No path exists between '{start}' and '{end}'")
                required.update(path)
            else:
                required.update([start, end])

        return required

    def _find_path(
        self, start: str, end: str, dependencies: Dict[str, Set[str]]
    ) -> Optional[List[str]]:
        """Find path between two nodes using BFS."""
        visited = set()
        queue = [(start, [start])]

        while queue:
            current, path = queue.pop(0)
            if current == end:
                return path

            for next_node in dependencies.get(current, set()):
                if next_node not in visited:
                    visited.add(next_node)
                    queue.append((next_node, path + [next_node]))

        return None

    def squash_into_subgraph(self, path: str, n_step_repeats: int = 1) -> Graph:
        """Squash a path into a single subgraph node."""
        subgraph = self.extract_path_as_graph(path, n_step_repeats)
        replaced_nodes = {node.external_name for node in subgraph.nodes}
        self.nodes = [
            node for node in self.nodes if node.external_name not in replaced_nodes
        ]
        self.insert_node(subgraph)
        return subgraph

    def __str__(self) -> str:
        """Return string representation of the graph."""
        nodes_str = "\n".join(
            f"{i:2d}. {node.__class__.__name__:<20} (root={node.is_root}, name={node.external_name})"
            for i, node in enumerate(self.nodes, 1)
        )
        return f"Graph with {len(self.nodes)} nodes:\n{nodes_str}"

    def detect_subgraphs(self) -> List[List[Node]]:
        """Detect independent subgraphs in the node network."""
        subgraphs: List[List[Node]] = []
        used_nodes = set()
        name_to_node = {node.external_name: node for node in self.nodes}

        # Build variable provider map
        providers = {
            f"{node.external_name}.{var.name}": node
            for node in self.nodes
            for var in node.variables
        }

        # Build dependency graph
        dependencies = self._build_bidirectional_dependencies(providers)

        # Find strongly connected components (feedback loops)
        sccs = self._find_strongly_connected_components(dependencies)

        # Add SCCs (feedback loops) first
        for scc in sccs:
            group = [name_to_node[name] for name in scc]
            subgraphs.append(group)
            used_nodes.update(scc)

        # Group remaining nodes
        remaining = [n for n in self.nodes if n.external_name not in used_nodes]

        # Add root nodes
        root_nodes = [
            n
            for n in remaining
            if not isinstance(n.inputs, Inputs) or not n.inputs.inputs
        ]
        if root_nodes:
            subgraphs.append(root_nodes)
            used_nodes.update(n.external_name for n in root_nodes)

        # Group nodes by shared dependencies
        remaining = [n for n in remaining if n.external_name not in used_nodes]
        while remaining:
            node = remaining[0]
            group = [node]
            node_deps = dependencies.get(node.external_name, set())

            for other in remaining[1:]:
                if dependencies.get(other.external_name, set()) == node_deps:
                    group.append(other)

            subgraphs.append(group)
            used_nodes.update(n.external_name for n in group)
            remaining = [n for n in remaining if n.external_name not in used_nodes]

        if self.debug:
            self._log_subgraph_analysis(subgraphs)

        return subgraphs

    def _build_bidirectional_dependencies(
        self, providers: Dict[str, Node]
    ) -> Dict[str, Set[str]]:
        """Build bidirectional dependency graph."""
        dependencies = {node.external_name: set() for node in self.nodes}

        for node in self.nodes:
            if isinstance(node.inputs, Inputs):
                for input_name in node.inputs.inputs:
                    if provider := providers.get(input_name):
                        dependencies[node.external_name].add(provider.external_name)
                        dependencies[provider.external_name].add(node.external_name)

        return dependencies

    def _find_strongly_connected_components(
        self, dependencies: Dict[str, Set[str]]
    ) -> List[List[str]]:
        """Find strongly connected components using Tarjan's algorithm."""
        index = {}
        lowlink = {}
        stack = []
        on_stack = set()
        sccs = []

        for node in self.nodes:
            if node.external_name not in index:
                _, new_sccs = find_scc(
                    node.external_name,
                    stack,
                    index,
                    lowlink,
                    on_stack,
                    0,
                    dependencies=dependencies,
                )
                sccs.extend(new_sccs)

        return sccs

    def _log_subgraph_analysis(self, subgraphs: List[List[Node]]) -> None:
        """Log detailed subgraph analysis."""
        logger.info("\nSubgraph Analysis:")
        for i, group in enumerate(subgraphs):
            logger.info(f"\nGroup {i}:")
            logger.info("  Nodes:")
            for node in group:
                logger.info(f"    {node.__class__.__name__}({node.external_name})")
            logger.info("  Consumes:")
            for node in group:
                if isinstance(node.inputs, Inputs):
                    logger.info(f"    {node.external_name}: {node.inputs.inputs}")
            logger.info("  Provides:")
            for node in group:
                logger.info(
                    f"    {node.external_name}: {[f'{node.external_name}.{v.name}' for v in node.variables]}"
                )

    def print_subgraphs(self, subgraphs: List[List[Node]]) -> None:
        """Print detected subgraphs in a readable format."""
        logger.info("\nDetected parallel execution groups:")
        for i, subgraph in enumerate(subgraphs, 1):
            nodes_str = "\n  ".join(
                f"- {node.__class__.__name__:<20} (root={node.is_root}, name={node.external_name})"
                for node in subgraph
            )
            logger.info(f"\nGroup {i}:\n  {nodes_str}")

    def analyze_group_dependencies(
        self, subgraphs: List[List[Node]]
    ) -> Dict[int, Set[int]]:
        """Analyze dependencies between groups."""
        var_to_group = {}
        group_contents = {}

        for group_idx, nodes in enumerate(subgraphs):
            group_contents[group_idx] = [node.__class__.__name__ for node in nodes]
            for node in nodes:
                for var in node.variables:
                    var_to_group[f"{node.external_name}.{var.name}"] = group_idx

                if isinstance(node.inputs, Inputs):
                    for input_name in node.inputs.inputs:
                        original_name = next(
                            (
                                old
                                for old, new in getattr(
                                    node, "_variable_mapping", {}
                                ).items()
                                if new == input_name
                            ),
                            input_name,
                        )
                        var_to_group[original_name] = group_idx

        group_deps = {i: set() for i in range(len(subgraphs))}

        if self.debug:
            self._log_group_dependencies(subgraphs, group_contents)

        for group_idx, nodes in enumerate(subgraphs):
            for node in nodes:
                if isinstance(node.inputs, Inputs):
                    for input_name in node.inputs.inputs:
                        if provider_group := var_to_group.get(input_name):
                            if provider_group != group_idx:
                                group_deps[group_idx].add(provider_group)

        return group_deps

    def _log_group_dependencies(
        self, subgraphs: List[List[Node]], group_contents: Dict[int, List[str]]
    ) -> None:
        """Log group dependency information."""
        logger.info("\nVariable providers and consumers:")
        for group_idx, nodes in enumerate(subgraphs):
            logger.info(
                f"\nGroup {group_idx} ({', '.join(group_contents[group_idx])}):"
            )
            for node in nodes:
                if isinstance(node.inputs, Inputs):
                    logger.info(
                        f"  {node.__class__.__name__} consumes: {node.inputs.inputs}"
                    )

    def reset(
        self,
        nodes_to_reset: Optional[List[Node]] = None,
        apply_reset_modifier: bool = False,
        apply_reset_modifier_to: Optional[List[Node]] = None,
    ) -> None:
        """Reset nodes in the graph."""
        target_nodes = nodes_to_reset if nodes_to_reset is not None else self.nodes
        apply_reset_modifier = apply_reset_modifier or apply_reset_modifier_to
        modifier_nodes = (
            apply_reset_modifier_to
            if apply_reset_modifier_to is not None
            else target_nodes
        )

        for node in target_nodes:
            node.reset(apply_reset_modifier=(node in modifier_nodes))

    def print_info(self, indent: int = 0) -> str:
        """Print detailed information about all nodes in the graph."""
        lines = []
        prefix = "  " * indent
        lines.append(f"{prefix}Graph: {self.external_name}")

        for node in self.nodes:
            if isinstance(node, Graph):
                lines.append(node.print_info(indent + 1))
            else:
                lines.append(f"{prefix}  Node: {node.__class__.__name__}")
                lines.append(f"{prefix}    External name: {node.external_name}")
                if isinstance(node.inputs, Inputs):
                    lines.append(f"{prefix}    Inputs: {node.inputs.inputs}")
                if node.resolved_inputs:
                    resolved = [var.full_name for var in node.resolved_inputs.inputs]
                    lines.append(f"{prefix}    Resolved inputs: {resolved}")

        msg = "\n".join(lines)
        if indent == 0:
            logger.info(msg)
        return msg

    def __deepcopy__(self, memo: Dict) -> Graph:
        """Custom deepcopy implementation to handle graph context."""
        # First, create a copy using parent's logic
        result: Graph = super().__deepcopy__(memo)  # type: ignore[assignment]

        # Clone nodes within the graph and store them in memo to avoid duplicates
        result.nodes = []
        node_mapping = {}  # Map old external names to new nodes
        internal_nodes = set()  # Track nodes that belong to this graph

        # First pass: clone all nodes and build mappings
        for node in self.nodes:
            if id(node) in memo:
                cloned_node = memo[id(node)]
            else:
                cloned_node = deepcopy(node, memo)
                memo[id(node)] = cloned_node
            result.nodes.append(cloned_node)
            node_mapping[node.external_name] = cloned_node
            internal_nodes.add(
                node.external_name.rsplit("_", 1)[0]
            )  # Base name without number

        # Second pass: update inputs for each node in the graph
        for node in result.nodes:
            if isinstance(node.inputs, Inputs):
                new_inputs = []
                for input_name in node.inputs.inputs:
                    provider_name, var_name = input_name.split(".")
                    provider_base = provider_name.rsplit("_", 1)[0]

                    # If provider was in our original graph, use the new mapped name
                    if provider_name in node_mapping:
                        new_inputs.append(
                            f"{node_mapping[provider_name].external_name}.{var_name}"
                        )
                    # If provider base name matches one of our internal nodes, update the number
                    elif provider_base in internal_nodes:
                        new_provider_name = f"{provider_base}_{len(node_mapping)}"
                        new_inputs.append(f"{new_provider_name}.{var_name}")
                    # Otherwise keep the external reference as is
                    else:
                        new_inputs.append(input_name)
                node.inputs = Inputs(new_inputs)

        # Re-collect node data for the graph
        result._collect_node_data()

        return result


class StepModifier(ABC):
    """Abstract base class for step function modifiers."""

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        pass

    def bind_to_node(self, node: Node) -> None:
        """Bind this modifier to a node's step and reset methods."""
        node.step = self.__call__.__get__(self, self.__class__)

        def make_reset(orig_reset, modifier):
            def reset_with_modifier(self, *args, **kwargs):
                orig_reset(*args, **kwargs)
                modifier.reset()

            return reset_with_modifier

        node.reset = make_reset(node.reset, self).__get__(node, node.__class__)


class ZeroOrderHold(StepModifier):
    def __init__(self, node: Node, clock_ref):
        self.node = node
        self.step_function = node.step
        self.clock = clock_ref
        self.last_update_time = None

    def __call__(self, *args, **kwargs):
        if (
            self.last_update_time is None
            or self.last_update_time + self.node.step_size <= self.clock.time.value
        ):
            self.step_function(*args, **kwargs)
            self.last_update_time = self.clock.time.value

    def reset(self):
        self.last_update_time = None


class ResetModifier(StepModifier):
    def __init__(self, node: Node, reset_semaphore: Node):
        self.node = node
        self.step_function = node.step
        self.reset_semaphore = reset_semaphore

    def __call__(self, *args, **kwargs):
        if self.reset_semaphore.reset_flag.value:
            self.node.reset()
        self.step_function(*args, **kwargs)

    def reset(self):
        pass
