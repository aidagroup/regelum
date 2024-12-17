"""Base classes for node-based computation."""

from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypedDict,
    TYPE_CHECKING,
)

import casadi as cs
import numpy as np
import torch

from regelum import _SYMBOLIC_INFERENCE_ACTIVE
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


class Metadata(TypedDict, total=False):
    """Metadata for a variable."""

    reset_modifier: Optional[Callable[..., Any]]
    initial_value: Any
    symbolic_value: Optional[cs.MX]
    shape: Optional[Tuple[int, ...]]


@dataclass(slots=True)
class Variable:
    """A variable with optional metadata."""

    name: str
    value: Optional[Any] = None
    metadata: Metadata = field(default_factory=Metadata)
    node_name: str = field(default="")

    def __post_init__(self) -> None:
        if "initial_value" not in self.metadata:
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
                self.metadata["symbolic_value"] = cs.MX.sym(self.name, *shape)
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
        var = Variable(
            name, value, metadata or Metadata(shape=shape), self.external_name
        )
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

    def resolve(self, variables: List[Variable]) -> None:
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

    def __or__(self, other: Node) -> Graph:
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

        # Copy attributes
        for k, v in self.__dict__.items():
            if k == "nodes" and isinstance(self, Graph):
                nodes_copy = [deepcopy(node, memo) for node in v]
                setattr(result, k, nodes_copy)
            elif k == "_variables":
                vars_copy = [deepcopy(var, memo) for var in v]
                setattr(result, k, vars_copy)
            elif k == "resolved_inputs":
                # Skip resolved_inputs - they will be re-resolved after cloning
                setattr(result, k, None)
            else:
                setattr(result, k, deepcopy(v, memo))

        # Handle instance counting and naming
        if not isinstance(self, Graph):
            cls._instances.setdefault(cls.__name__, []).append(result)
            result._external_name = (
                f"{result._internal_name}_{len(cls._instances[cls.__name__])}"
            )
            for var in result._variables:
                var.node_name = result._external_name
        else:
            result._external_name = (
                f"{result._internal_name}_{self.get_instance_count()}"
            )

        # Update input references if part of a graph
        if hasattr(result, "inputs") and isinstance(result.inputs, Inputs):
            parent_graph = memo.get("parent_graph")
            if parent_graph:
                graph_vars = {
                    f"{node.external_name}.{var.name}": node
                    for node in parent_graph.nodes
                    for var in node.variables
                }

                new_inputs = []
                for input_name in result.inputs.inputs:
                    if input_name in graph_vars:
                        provider_node = graph_vars[input_name]
                        provider_name, var_name = input_name.split(".")
                        new_inputs.append(f"{provider_node.external_name}.{var_name}")
                    else:
                        new_inputs.append(input_name)
                result.inputs = Inputs(new_inputs)

        return result


class Graph(Node):
    """A graph of connected nodes."""

    def __init__(
        self, nodes: List[Node], debug: bool = False, n_step_repeats: int = 1
    ) -> None:
        super().__init__(name="graph")
        self.nodes = nodes
        self.debug = debug
        self.n_step_repeats = n_step_repeats
        self._collect_node_data()
        self.resolve_status = ResolveStatus.UNDEFINED

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

        # Collect resolved inputs if any node has them
        if any(node.resolved_inputs for node in self.nodes):
            resolved_vars = [
                var
                for node in self.nodes
                if node.resolved_inputs
                for var in node.resolved_inputs.inputs
            ]
            self.resolved_inputs = ResolvedInputs(resolved_vars)

        # Collect all variables
        self._variables = [var for node in self.nodes for var in node.variables]

    def parallelize(self) -> ParallelGraph:
        """Convert to parallel execution mode."""
        from regelum.environment.node.parallel import ParallelGraph

        return ParallelGraph(self.nodes, self.debug)

    def step(self) -> None:
        """Execute all nodes in sequence."""
        for _ in range(self.n_step_repeats):
            for node in self.nodes:
                node.step()

    def resolve(self, variables: List[Variable]) -> None:
        """Resolve inputs for all nodes and determine execution order."""
        # Check for duplicate variable names
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

        # Build dependency graph for execution order
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

        # Sort nodes by dependencies
        ordered_nodes = self._sort_nodes_by_dependencies(dependencies)
        self.nodes = ordered_nodes

        if self.debug:
            logger.info(f"\nResolved node execution order:\n{self}")

        # Resolve inputs for all nodes
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

        # Process root nodes first
        ordered.extend(root_nodes)
        visited.update(node.external_name for node in root_nodes)

        # Process remaining nodes
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
            node.resolve(node.variables + self.variables)
        return self.resolve(self.variables)

    def clone_node(self, node_name: str, new_name: Optional[str] = None) -> Node:
        """Clone a node and its variables, then insert it into the graph."""
        original_node = next(
            (node for node in self.nodes if node.external_name == node_name), None
        )
        if not original_node:
            available = "\n- ".join(node.external_name for node in self.nodes)
            raise ValueError(
                f"Node '{node_name}' not found in graph.\nAvailable nodes:\n- {available}"
            )

        memo = {"parent_graph": self}
        cloned_node = deepcopy(original_node, memo)

        if new_name is not None:
            cloned_node.alter_name(new_name)

        if isinstance(cloned_node, Graph):
            graph_num = int(cloned_node.external_name.split("_")[-1])

            # Create mapping for all nodes in the cloned graph
            node_mapping = {}
            cloned_node_names = set()
            for node in cloned_node.nodes:
                old_name = node.external_name
                base_name = "_".join(old_name.split("_")[:-1])
                new_name = f"{base_name}_{graph_num}"
                node_mapping[old_name] = new_name
                cloned_node_names.add(base_name)
                node.alter_name(base_name)

            # Update input references for all nodes
            def update_node_inputs(node: Node) -> None:
                if isinstance(node.inputs, Inputs):
                    new_inputs = []
                    for input_name in node.inputs.inputs:
                        provider_name, var_name = input_name.split(".")
                        provider_base = "_".join(provider_name.split("_")[:-1])

                        if provider_name in node_mapping:
                            new_provider_name = node_mapping[provider_name]
                            new_inputs.append(f"{new_provider_name}.{var_name}")
                        elif provider_base in cloned_node_names:
                            new_inputs.append(f"{provider_base}_{graph_num}.{var_name}")
                        else:
                            new_inputs.append(input_name)
                    node.inputs = Inputs(new_inputs)
                    # Clear resolved inputs to force re-resolution
                    node.resolved_inputs = None

            # Update inputs for all nodes in the cloned graph
            for node in cloned_node.nodes:
                update_node_inputs(node)

            # Re-collect node data and resolve the cloned graph

            cloned_node.resolve(cloned_node.variables)
            cloned_node._collect_node_data()

            # Ensure each node in the cloned graph is resolved
            for node in cloned_node.nodes:
                if not node.is_resolved:
                    node.resolve(cloned_node.variables)

        # Insert into parent graph and resolve
        self.insert_node(cloned_node)
        self.resolve(self.variables)

        return cloned_node

    def extract_path_as_graph(self, path: str, n_step_repeats: int = 1) -> Graph:
        """Extract a minimal subgraph containing specified nodes and necessary interim nodes.

        Args:
            path: String in format "node1 -> node2 -> node3" specifying desired path
            n_step_repeats: Number of times to repeat the step operation for the subgraph
        Returns:
            Graph containing the minimal required subgraph

        Raises:
            ValueError: If path format is invalid or nodes can't be found
        """
        # Validate and parse path
        if not path or "->" not in path:
            raise ValueError(
                "Path must be in format: 'node1 -> node2 -> node3'. " f"Got: '{path}'"
            )

        node_names = [name.strip() for name in path.split("->")]
        if not all(node_names):
            raise ValueError("Empty node names are not allowed. " f"Path: '{path}'")

        # Map of node name -> Node instance
        name_to_node = {node.external_name: node for node in self.nodes}

        # Verify all nodes exist
        missing_nodes = [name for name in node_names if name not in name_to_node]
        if missing_nodes:
            available = "\n- ".join(sorted(name_to_node.keys()))
            raise ValueError(
                f"Could not find nodes: {missing_nodes}\n"
                f"Available nodes:\n- {available}"
            )

        # Build dependency graph (both input dependencies and usage dependencies)
        dependencies: Dict[str, Set[str]] = {
            node.external_name: set() for node in self.nodes  # Initialize all nodes
        }

        for node in self.nodes:
            # Add input dependencies
            if isinstance(node.inputs, Inputs):
                for input_name in node.inputs.inputs:
                    for provider in self.nodes:
                        if any(
                            input_name == f"{provider.external_name}.{var.name}"
                            for var in provider.variables
                        ):
                            dependencies[node.external_name].add(provider.external_name)
                            dependencies[provider.external_name].add(node.external_name)
                            break

            # Add usage dependencies - who uses this node's variables
            for var in node.variables:
                var_name = f"{node.external_name}.{var.name}"
                for consumer in self.nodes:
                    if isinstance(consumer.inputs, Inputs):
                        if var_name in consumer.inputs.inputs:
                            dependencies[node.external_name].add(consumer.external_name)
                            dependencies[consumer.external_name].add(node.external_name)

        # Find all required nodes between each consecutive pair
        required_nodes = set()
        for i in range(len(node_names) - 1):
            start, end = node_names[i], node_names[i + 1]

            # Check if direct path exists
            if end not in dependencies.get(start, set()):
                # Try to find path through other nodes
                path_found = False
                visited = set()
                to_visit = [(start, [start])]

                while to_visit and not path_found:
                    current, path_so_far = to_visit.pop(0)
                    if current == end:
                        required_nodes.update(path_so_far)
                        path_found = True
                        continue

                    for next_node in dependencies.get(current, set()):
                        if next_node not in visited:
                            visited.add(next_node)
                            to_visit.append((next_node, path_so_far + [next_node]))

                if not path_found:
                    raise ValueError(
                        f"No path exists between '{start}' and '{end}'. "
                        "Nodes are not connected in the dependency graph."
                    )
            else:
                required_nodes.update([start, end])

        # Create subgraph with required nodes
        subgraph_nodes = [
            node for node in self.nodes if node.external_name in required_nodes
        ]

        return Graph(subgraph_nodes, debug=self.debug, n_step_repeats=n_step_repeats)

    def squash_into_subgraph(self, path: str, n_step_repeats: int = 1) -> None:
        """Squash a path into a single node, replacing original nodes with a subgraph.

        Args:
            path: String in format "node1 -> node2 -> node3" specifying path to squash
            n_step_repeats: Number of times to repeat the step operation for the squashed subgraph
        """
        # Extract subgraph
        subgraph = self.extract_path_as_graph(path, n_step_repeats)

        # Get names of nodes to replace
        replaced_nodes = {node.external_name for node in subgraph.nodes}

        # Remove original nodes from the graph
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

        # Build dependency graph
        dependencies: Dict[str, Set[str]] = {}
        provides: Dict[str, Node] = {}

        # Map which nodes provide which variables
        for node in self.nodes:
            for var in node.variables:
                var_name = f"{node.external_name}.{var.name}"
                provides[var_name] = node

        # Build dependency map
        for node in self.nodes:
            if isinstance(node.inputs, Inputs):
                for input_name in node.inputs.inputs:
                    if provider := provides.get(input_name):
                        dependencies.setdefault(node.external_name, set()).add(
                            provider.external_name
                        )
                        dependencies.setdefault(provider.external_name, set())

        # Find strongly connected components (feedback loops)

        # Find all strongly connected components
        index: Dict[str, int] = {}
        lowlink: Dict[str, int] = {}
        stack: List[str] = []
        on_stack: Set[str] = set()
        sccs: List[List[str]] = []

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

        # Convert SCCs to node groups
        name_to_node = {node.external_name: node for node in self.nodes}
        used_nodes = set()

        # First add SCCs (feedback loops)
        for scc in sccs:
            group = [name_to_node[name] for name in scc]
            subgraphs.append(group)
            used_nodes.update(scc)

        # Group remaining nodes by their dependencies
        remaining_nodes = [n for n in self.nodes if n.external_name not in used_nodes]

        # Find nodes with no inputs (root nodes)
        root_nodes = [
            n
            for n in remaining_nodes
            if not isinstance(n.inputs, Inputs) or not n.inputs.inputs
        ]
        if root_nodes:
            subgraphs.append(root_nodes)
            used_nodes.update(n.external_name for n in root_nodes)

        # Group remaining nodes based on shared dependencies
        remaining = [n for n in remaining_nodes if n.external_name not in used_nodes]
        while remaining:
            node = remaining[0]
            group = [node]
            node_deps = dependencies.get(node.external_name, set())

            # Find nodes with similar dependencies
            for other in remaining[1:]:
                other_deps = dependencies.get(other.external_name, set())
                if node_deps == other_deps:
                    group.append(other)

            subgraphs.append(group)
            used_nodes.update(n.external_name for n in group)
            remaining = [n for n in remaining if n.external_name not in used_nodes]

        if self.debug:
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

        return subgraphs

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
        # Build variable provider map
        var_to_group: Dict[str, int] = {}
        group_contents: Dict[int, List[str]] = {}

        # First pass - map all variables to their groups
        for group_idx, nodes in enumerate(subgraphs):
            group_contents[group_idx] = [node.__class__.__name__ for node in nodes]
            for node in nodes:
                # Map both original and renamed variables
                for var in node.variables:
                    var_to_group[f"{node.external_name}.{var.name}"] = group_idx

                # Map renamed inputs to their providers
                if hasattr(node, "inputs") and isinstance(node.inputs, Inputs):
                    for input_name in node.inputs.inputs:
                        # Find original variable name if this is a renamed input
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

        # Build group dependency map
        group_deps: Dict[int, Set[int]] = {i: set() for i in range(len(subgraphs))}

        # Debug print
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

        for group_idx, nodes in enumerate(subgraphs):
            for node in nodes:
                if isinstance(node.inputs, Inputs):
                    for input_name in node.inputs.inputs:
                        if provider_group := var_to_group.get(input_name):
                            if provider_group != group_idx:
                                group_deps[group_idx].add(provider_group)

        return group_deps

    def reset(
        self,
        nodes_to_reset: Optional[List[Node]] = None,
        apply_reset_modifier_to: Optional[List[Node]] = None,
    ) -> None:
        """Reset nodes in the graph.

        Args:
            nodes_to_reset: Specific nodes to reset. If None, resets all nodes
            apply_reset_modifier_to: Nodes that should use reset modifier. If None, applies to all reset nodes
        """
        target_nodes = nodes_to_reset if nodes_to_reset is not None else self.nodes
        modifier_nodes = (
            apply_reset_modifier_to
            if apply_reset_modifier_to is not None
            else target_nodes
        )

        for node in target_nodes:
            node.reset(apply_reset_modifier=(node in modifier_nodes))


class Clock(Node):
    """Time management node."""

    def __init__(self, fundamental_step_size: float) -> None:
        """Initialize the Clock node.

        Args:
            fundamental_step_size: Fundamental step size for time increments.
        """
        super().__init__(
            step_size=fundamental_step_size,
            is_continuous=False,
            is_root=False,
            name="clock",
        )
        self.fundamental_step_size = fundamental_step_size
        self.time = self.define_variable("time", value=0.0)

    def step(self) -> None:
        """Increment time by fundamental step size."""
        self.time.value += self.fundamental_step_size


class StepCounter(Node):
    """Counts steps in the simulation."""

    def __init__(self, nodes: List[Node], start_count: int = 0) -> None:
        """Initialize the StepCounter node.

        Args:
            nodes: List of nodes to track
            start_count: Initial counter value
        """
        # Find minimum step size among non-continuous nodes
        step_sizes = [node.step_size for node in nodes if not node.is_continuous]
        if not step_sizes:
            raise ValueError("No non-continuous nodes provided")
        min_step_size = min(step_sizes)

        super().__init__(
            step_size=min_step_size,
            is_continuous=False,
            is_root=False,
            name="step_counter",
        )

        self.counter = self.define_variable("counter", value=start_count)
        self.nodes = nodes

    def step(self) -> None:
        """Increment counter by 1."""
        self.counter.value += 1


class Logger(Node):
    """State recording node."""

    def __init__(
        self, variables_to_log: List[str], step_size: float, cooldown: float = 0.0
    ) -> None:
        """Initialize the Logger node."""
        super().__init__(
            inputs=["clock_1.time", "step_counter_1.counter"] + variables_to_log,
            step_size=step_size,
            is_continuous=False,
            is_root=False,
            name="logger",
        )

        self.variables_to_log = variables_to_log
        self.cooldown = cooldown
        self.last_log_time = self.define_variable("last_log_time", value=-float("inf"))
        self.log_queue: Optional[multiprocessing.Queue] = None

    def step(self) -> None:
        """Log current state if enough time has passed."""
        current_time = self.resolved_inputs.find("clock.time").value

        if current_time - self.last_log_time.value >= self.cooldown:
            log_parts = [f"t={current_time:.3f}"]

            for path in self.inputs.inputs:
                value = self.resolved_inputs.find(path).value
                if isinstance(value, (np.ndarray, list)):
                    formatted_value = f"[{', '.join(f'{v:.3f}' for v in value)}]"
                else:
                    formatted_value = f"{value:.3f}"
                log_parts.append(f"{path}={formatted_value}")

            log_msg = f"{self.external_name} | " + " | ".join(log_parts)

            if self.log_queue is not None:
                self.log_queue.put(log_msg)
            else:
                logger.info(log_msg)

            self.last_log_time.value = current_time
