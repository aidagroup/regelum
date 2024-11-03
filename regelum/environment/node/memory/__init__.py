from regelum.environment.node.base import Node, State
from typing import List, Optional, Dict, Any, Sequence


class MemoryCell(Node):
    def __init__(
        self,
        node_to_remember: Node,
        paths_to_remember: Optional[List[str]] = None,
        step_size: float = None,
        prefix: Optional[str] = None,
    ):
        self.node_to_remember = node_to_remember
        if paths_to_remember is None:
            paths_to_remember = [node_to_remember.state.name]
        self.paths_to_remember = paths_to_remember

        # Extract leaf names from paths
        leaf_names = [path.split("/")[-1] for path in paths_to_remember]

        # Create a state with two substates using leaf names
        state = State(
            "memory",
            None,
            [
                self.node_to_remember.state,
                State(
                    "current",
                    None,
                    [
                        State(
                            name,
                            self.node_to_remember.state[path].shape,
                            self.node_to_remember.state[path].data,
                        )
                        for name, path in zip(leaf_names, paths_to_remember)
                    ],
                ),
                State(
                    "previous",
                    None,
                    [
                        State(
                            name,
                            self.node_to_remember.state[path].shape,
                            self.node_to_remember.state[path].data,
                        )
                        for name, path in zip(leaf_names, paths_to_remember)
                    ],
                ),
            ],
        )

        inputs = node_to_remember.state.paths
        super().__init__(step_size=step_size, state=state, inputs=inputs, prefix=prefix)

    def compute_state_dynamics(self) -> Dict[str, Any]:
        # Move current state to previous
        for path in self.paths_to_remember:
            leaf_name = path.split("/")[-1]
            current_state = self.state[f"memory/current/{leaf_name}"].data
            self.state[f"memory/previous/{leaf_name}"].data = current_state

        # Update current state with new values
        new_values = {}
        for path in self.paths_to_remember:
            leaf_name = path.split("/")[-1]
            new_values[f"memory/current/{leaf_name}"] = self.node_to_remember.state[
                path
            ].data

        return new_values


def create_memory_chain(
    target_node: Node,
    n_cells: int,
    paths_to_remember: Sequence[str],
    step_size: float,
    prefix: Optional[str] = None,
) -> List[MemoryCell]:
    """Create a chain of memory cells that remember n last states of target node. Returns list of cells ordered from newest to oldest memory."""
    cells: List[MemoryCell] = []
    prefixes = (
        [f"{prefix}_{i}" for i in range(1, n_cells + 1)]
        if prefix is not None
        else list(range(1, n_cells + 1))
    )
    # First cell remembers target node
    cells.append(
        MemoryCell(target_node, paths_to_remember, step_size, prefix=prefixes[0])
    )

    # Each subsequent cell remembers previous cell's state
    for i in range(n_cells - 1):
        prev_cell = cells[-1]
        remembered_paths = [
            f"{prefixes[i]}_memory/previous/{path.split('/')[-1]}"
            for path in paths_to_remember
        ]
        cells.append(
            MemoryCell(prev_cell, remembered_paths, step_size, prefix=prefixes[i + 1])
        )

    return cells
