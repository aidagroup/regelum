"""Graph class for managing node execution order and dependencies."""

from __future__ import annotations
import numpy as np
from typing import List, Optional
from functools import reduce
from math import gcd
from graphviz import Digraph
from copy import deepcopy
from regelum.environment.node.base import Node, State, Inputs
from regelum.environment.transistor import SampleAndHoldModifier
from regelum.environment.node.base import Logger
from regelum.environment.node.base import Clock
from regelum.environment.node.base import StepCounter


def visualize_graph(
    graph: "Graph", output_file: Optional[str] = "graph_diagram", view: bool = True
):
    """Visualize the given Graph instance as a block diagram using graphviz.

    Each Node is represented as a box with:
      - "ClassName:root_state_name" as title (bold)
      - Node type (Continuous or Discrete)
      - Initial state values (if leaf and defined)
      - A table showing the transistor and its modifiers (if any).
        Modifiers are detected by a class attribute `is_modifier = True`.
        The main transistor class is one of ODETransistor, ScipyTransistor, CasADiTransistor,
        or Transistor if none of these specialized classes appear.

    Edges represent dependencies between nodes based on their input states and are labeled:
      - The labels are bold, slightly larger, with a white background.
    """
    dot = Digraph(comment="Regelum Graph Visualization", format="pdf")
    dot.attr(rankdir="LR", concentrate="true", nodesep="0.5", ranksep="0.7")

    nodes = graph.ordered_nodes
    node_ids = {node: f"node_{i}" for i, node in enumerate(nodes)}

    # Track instance counts for each class
    class_counts = {}

    # First pass to count instances
    for node in nodes:
        class_name = node.__class__.__name__
        class_counts[class_name] = class_counts.get(class_name, 0) + 1

    # Separate counters for nodes and edges
    node_instance_counters = {}
    edge_instance_counters = {}

    def get_instance_label(class_name: str, is_edge: bool = False) -> str:
        if class_counts[class_name] == 1:
            return class_name

        counters = edge_instance_counters if is_edge else node_instance_counters
        counters[class_name] = counters.get(class_name, 0) + 1
        return f"{class_name}_{counters[class_name]}"

    def prettify_value(val):
        if isinstance(val, np.ndarray):
            return f"{val}"
        elif hasattr(val, "__class__"):
            return val.__class__.__name__
        else:
            return str(val)

    def get_transistor_info(transistor):
        if not transistor:
            return None, [], False

        mro = type(transistor).mro()
        modifiers = []
        main_transistor_cls = None
        has_reset = False

        # First pass - collect all modifiers and check for reset
        for cls in mro:
            if cls is object:
                continue
            if getattr(cls, "is_modifier", False):
                if cls.__name__ == "ResetTransistor":
                    has_reset = True
                else:
                    modifiers.append(cls.__name__)

        # Second pass - find main transistor class
        for cls in mro:
            if cls is object or cls.__name__ == "Transistor":
                continue
            if not getattr(cls, "is_modifier", False):  # Only consider non-modifiers
                main_transistor_cls = cls.__name__
                break

        if main_transistor_cls is None:
            main_transistor_cls = "Transistor"

        modifiers.reverse()
        return main_transistor_cls, modifiers if modifiers else None, has_reset

    # Render each node
    for node in nodes:
        node_id = node_ids[node]
        state_name = node.state.name
        instance_label = get_instance_label(node.__class__.__name__)
        node_type_info = (
            "Transition: Continuous" if node.is_continuous else "Transition: Discrete"
        )

        init_val_str = ""
        if node.state.is_leaf and node.state.data is not None:
            val_str = prettify_value(node.state.data)
            init_val_str = f"Init state: {val_str}"

        main_transistor_cls, modifiers, has_reset = get_transistor_info(node.transistor)

        # Start main table
        label = """<<table border='0' cellborder='0' cellspacing='0'>"""

        label += f"""<tr><td><b>{instance_label}:{state_name}</b></td></tr>
            <tr><td>{node_type_info}</td></tr>"""

        if init_val_str:
            label += f"<tr><td>{init_val_str}</td></tr>"

        if main_transistor_cls:
            label += "<tr><td><table border='1' cellborder='1' cellspacing='0'>"
            if modifiers:
                for modifier in modifiers:
                    label += f"<tr><td bgcolor='#d9d9d9'>{modifier}</td></tr>"
            label += f"<tr><td bgcolor='#e6e6e6'>{main_transistor_cls}</td></tr>"
            label += "</table></td></tr>"

        # Add arrow-like reset indicator at the bottom if needed
        if has_reset:
            label += """<tr><td align="center">
                <table border='0' cellborder='0' cellpadding='2'>
                    <tr>
                        <td bgcolor="#ffcccc" border='1' sides="lt" style="rounded"><font point-size="12"><b>➜ Reset</b></font></td>
                    </tr>
                </table>
            </td></tr>"""

        label += "</table>>"

        dot.node(
            node_id,
            label=label,
            shape="box",
            style="filled",
            fillcolor="#f0f0f0",
            margin="0.3",
        )

    # Map full_path -> list of owner nodes (instead of single owner)
    state_to_owners = {}
    for n in nodes:
        for s in n.state.get_all_states():
            if s.is_leaf:
                full_path = s.paths[0]
                if full_path not in state_to_owners:
                    state_to_owners[full_path] = []
                state_to_owners[full_path].append(n)

    # Track producer labels for each class (not per path)
    producer_labels = {}

    def get_producer_label(node):
        class_name = node.__class__.__name__
        if class_name not in producer_labels:
            producer_labels[class_name] = {}
        if node not in producer_labels[class_name]:
            count = len(producer_labels[class_name]) + 1
            producer_labels[class_name][node] = (
                class_name if class_counts[class_name] == 1 else f"{class_name}_{count}"
            )
        return producer_labels[class_name][node]

    # Add edges with consistent producer labels
    for node in nodes:
        for input_state in node.inputs.states:
            if input_state.is_leaf:
                input_path = input_state.paths[0]
                producers = state_to_owners.get(input_path, [])
                for producer in producers:
                    if producer != node:
                        producer_label = get_producer_label(producer)
                        # Use the last part of the path for nested states
                        state_name = input_path.split("/")[-1]
                        state_label = (
                            f"{producer_label}:{state_name}"
                            if len(producers) > 1
                            else state_name
                        )

                        edge_label = f"""<<table border='0' cellborder='0'>
                            <tr><td bgcolor='white'><font point-size='12'><b>{state_label}</b></font></td></tr>
                            </table>>"""

                        dot.edge(
                            node_ids[producer],
                            node_ids[node],
                            label=edge_label,
                            fontsize="10",
                            labeldistance="0.2",
                            labelangle="0",
                            labelfloat="false",
                        )

    dot.render(output_file, view=view)


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
        self.link_inputs(self.nodes)
        self.ordered_nodes = self.resolve(self.nodes)
        self._log_node_order()
        self.apply_transistors(self.ordered_nodes)

    def link_inputs(self, nodes: List[Node]):
        states = reduce(
            lambda x, y: x + y, [node.state.get_all_states() for node in nodes]
        )

        for node in nodes:
            node.inputs.resolve(states)

    def apply_transistors(self, nodes: List[Node]):
        from regelum.environment.transistor import ResetModifier

        reset_map = {}
        for node in self.nodes:
            for input_path in node.state.paths:
                if input_path.startswith("reset_"):

                    target_node_name = input_path[6:]  # Remove 'reset_' prefix
                    reset_map[target_node_name] = True

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
        for node in nodes:
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
        # 1. Collect all full paths from each node and ensure uniqueness
        full_path_to_node = {}
        for node in nodes:
            for path in node.state.paths:
                if path in full_path_to_node:
                    # Handle the case where different nodes produce the same full path
                    # This is not allowed and should not be allowed
                    other_node = full_path_to_node[path]
                    raise ValueError(
                        f"Duplicate full state path detected: '{path}' is produced by both "
                        f"'{other_node.state.name}' and '{node.state.name}'. "
                        "All full state paths must be unique."
                    )
                full_path_to_node[path] = node

        # 2. Build the adjacency list based on full paths
        graph = {node: [] for node in nodes}

        # For each node, resolve its inputs using full paths
        # If any input does not map to an existing path, raise an error
        for node in nodes:
            if not node.inputs.paths_to_states:
                continue
            for input_path in node.inputs.paths_to_states:
                if input_path not in full_path_to_node:
                    raise ValueError(
                        f"Input path '{input_path}' required by '{node.state.name}' "
                        "does not map to any known node's full state path."
                    )
                producer_node = full_path_to_node[input_path]
                if producer_node != node:
                    graph[producer_node].append(node)

        # 3. Detect strongly connected components (SCCs) to handle cycles as blocks
        index = 0
        stack = []
        on_stack = set()
        indices = {}
        low_link = {}
        sccs = []

        def strongconnect(v):
            nonlocal index
            indices[v] = index
            low_link[v] = index
            index += 1
            stack.append(v)
            on_stack.add(v)

            for w in graph[v]:
                if w not in indices:
                    strongconnect(w)
                    low_link[v] = min(low_link[v], low_link[w])
                elif w in on_stack:
                    low_link[v] = min(low_link[v], indices[w])

            # If v is the root of an SCC
            if low_link[v] == indices[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    scc.append(w)
                    if w == v:
                        break
                sccs.append(scc)

        for n in nodes:
            if n not in indices:
                strongconnect(n)

        # 4. Build SCC graph (condensation)
        node_to_scc = {}
        for i, scc in enumerate(sccs):
            for node in scc:
                node_to_scc[node] = i

        scc_graph = {i: set() for i in range(len(sccs))}
        for u in nodes:
            u_scc = node_to_scc[u]
            for v in graph[u]:
                v_scc = node_to_scc[v]
                if u_scc != v_scc:
                    scc_graph[u_scc].add(v_scc)

        # 5. Topological sort on scc_graph
        in_degree = {i: 0 for i in range(len(sccs))}
        for u in scc_graph:
            for v in scc_graph[u]:
                in_degree[v] += 1

        # Sort SCCs by lexicographically smallest state name for stable order
        def scc_key(scc_index):
            names = [node.state.name for node in sccs[scc_index]]
            return min(names)

        ready = [i for i in in_degree if in_degree[i] == 0]
        ready.sort(key=scc_key)

        scc_order = []
        while ready:
            current = ready.pop(0)
            scc_order.append(current)
            for dep in scc_graph[current]:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    ready.append(dep)
            ready.sort(key=scc_key)

        # 6. Expand each SCC. Sort nodes within each SCC block by state name, prioritizing root nodes.
        final_order = []
        for scc_idx in scc_order:
            scc_block = sccs[scc_idx]
            if len(scc_block) > 1 or any(n in graph[n] for n in scc_block):
                # Cycle or self-loop
                # Sort by is_root first (True before False), then by state name
                scc_block.sort(
                    key=lambda x: (not getattr(x, "is_root", False), x.state.name)
                )
            else:
                # Single node, no cycle
                # Same sorting strategy for consistency
                scc_block.sort(
                    key=lambda x: (not getattr(x, "is_root", False), x.state.name)
                )

            final_order.extend(scc_block)

        return final_order

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

    def extract_subgraph(self, path_expression: str, freezed=None):
        """Extract a subgraph based on a path expression like "pendulum_state -> controller_state".

        This involves:
          - Parsing the expression to identify start and end nodes/states.
          - Finding all nodes required to produce that end state from the start state.
          - If some inputs come from nodes not included, mark them as frozen.

        Returns a LazySubgraph that represents this subgraph.
        """
        # Parse path_expression (e.g., "pendulum_state -> controller_state")
        start_name, end_name = [x.strip() for x in path_expression.split("->")]

        # Identify the nodes corresponding to these states
        start_node = self._find_node_by_state_name(start_name)
        end_node = self._find_node_by_state_name(end_name)

        # Determine the set of nodes needed to produce end_node's inputs from start_node
        needed_nodes, frozen_inputs = self._resolve_subgraph_nodes(
            start_node, end_node, freezed
        )

        return LazySubgraph(needed_nodes, frozen_inputs=frozen_inputs)

    def _find_node_by_state_name(self, state_name: str):
        for node in self.nodes:
            if node.state.name == state_name:
                return node
        raise ValueError(f"No node found for state name '{state_name}'")

    def _resolve_subgraph_nodes(self, start_node, end_node, freezed):
        """Figure out which nodes are needed to get from start_node to end_node.

        Only includes nodes that are part of the minimal path between start and end,
        considering freezed inputs.
        """
        freezed = set(freezed or [])
        needed = set()
        queue = [(end_node, set())]  # (node, visited_paths)
        external_inputs = {}
        path_to_producer = {}

        # First pass: build path_to_producer map
        for node in self.nodes:
            for state in node.state.get_all_states():
                if state.is_leaf:
                    for path in state.paths:
                        path_to_producer[path] = node

        # Second pass: traverse backwards, tracking visited paths
        while queue:
            current, visited_paths = queue.pop(0)
            if current not in needed:
                needed.add(current)

                # Process inputs
                for input_state in current.inputs.states:
                    input_path = input_state.paths[0]

                    # Skip if we've seen this path or it's freezed
                    if input_path in visited_paths or input_path in freezed:
                        continue

                    producer_node = path_to_producer.get(input_path)
                    if producer_node is None or producer_node not in self.nodes:
                        # External input - freeze it
                        external_inputs[input_path] = input_state.data
                    else:
                        # Add path to visited and continue traversal
                        new_visited = visited_paths | {input_path}
                        queue.append((producer_node, new_visited))

        # Remove nodes not in the path from start to end
        reachable = set()
        queue = [(start_node, {start_node})]
        while queue:
            current, visited = queue.pop(0)
            for node in needed:
                if node not in visited:
                    # Check if current connects to node through needed nodes
                    for input_state in node.inputs.states:
                        producer = path_to_producer.get(input_state.paths[0])
                        if producer == current:
                            reachable.update(visited)
                            queue.append((node, visited | {node}))
                            break

        # Final set is intersection of needed and reachable nodes
        minimal_set = (reachable & needed) | {start_node, end_node}

        return list(minimal_set), external_inputs

    def _find_producer_of_state(self, input_state):
        # find which node produces input_state
        for node in self.nodes:
            if any(path == input_state.paths[0] for path in node.state.paths):
                return node
        return None

    def insert(self, subgraphs: List[LazySubgraph]):
        """Insert subgraphs back into the main graph.

        This means:
          - Add their nodes to self.nodes
          - Resolve any frozen inputs by linking them to actual states/nodes if now available
          - Possibly rename states and ensure uniqueness
        """
        for sg in subgraphs:
            for node in sg.nodes:
                # Check for name collisions, rename states if needed
                self.ordered_nodes.append(node)
            # Handle frozen inputs if we now have a place for them, or keep them as constants

        self.link_inputs(self.ordered_nodes)
        self.ordered_nodes = self.resolve(self.ordered_nodes)
        self._log_node_order()
        self.apply_transistors(self.ordered_nodes)
        self.link_inputs(self.ordered_nodes)


class LazySubgraph:
    """Represents a lazy subgraph with nodes and optional frozen inputs."""

    def __init__(self, nodes, frozen_inputs=None):
        """Initialize the LazySubgraph.

        Args:
            nodes: List of nodes constituting the subgraph.
            frozen_inputs: A dict of {full_path: value} for inputs that are considered external/frozen.
        """
        self.nodes = nodes[:]
        self.frozen_inputs = frozen_inputs or {}
        # Simple map of node.state.name to node for convenience
        self.node_map = {node.state.name: node for node in self.nodes}

    def multiply(self, n_copies: int):
        """Create multiple copies of this subgraph, each with unique suffixes."""
        subgraphs = []
        for i in range(n_copies):
            new_nodes = []
            suffix = f"_sub_{i}"
            for node in self.nodes:
                new_node = self._deepcopy_node_with_suffix(node, suffix)
                new_nodes.append(new_node)
            # Frozen inputs remain the same
            new_sg = LazySubgraph(new_nodes, frozen_inputs=deepcopy(self.frozen_inputs))
            subgraphs.append(new_sg)
        return subgraphs

    def attach(self, new_node):
        """Attach a new node to the subgraph."""
        self.nodes.append(new_node)
        self.node_map[new_node.state.name] = new_node

    def select_nodes_contain(self, substr_list: List[str]):
        """Select nodes whose state names contain any of the given substrings."""
        selected = []
        for node in self.nodes:
            if any(s in node.state.name for s in substr_list):
                selected.append(node)
        return selected

    def _deepcopy_node_with_suffix(self, node, suffix: str):
        # Deep copy node properly and rename its states and inputs
        new_node: Node = self._deepcopy_node(node)
        # Append suffix to state name and paths
        new_node.state.name += suffix
        new_node.state._validate_hierarchical_state()
        new_node.state._build_path_cache()
        # Update input paths if they refer to internal states
        # If the input path refers to states that are within this subgraph, add suffix
        new_paths = []
        for p in new_node.inputs.paths_to_states:
            # Check if p belongs to any original node's state
            if self._path_in_subgraph(p):
                p_split = p.split("/")
                p_split[0] = p_split[0] + suffix
                p = "/".join(p_split)
            new_paths.append(p)
        new_node.inputs.paths_to_states = new_paths
        return new_node

    def _deepcopy_node(self, node: Node):
        # Create a deep copy of node, ensuring states and inputs are also deep-copied
        # Assuming node, node.state, node.inputs, etc. can be copied safely.
        # If more complex logic is needed, implement it here.
        new_state = deepcopy(node.state)
        new_inputs = Inputs(node.inputs.paths_to_states)
        new_node = deepcopy(node)
        # new_node.transistor = None
        new_node.state = new_state
        new_node.inputs = new_inputs
        return new_node

    def _path_in_subgraph(self, path: str) -> bool:
        # Check if this path is produced by a node in this subgraph
        for n in self.nodes:
            if path in n.state.paths:
                return True
        return False
