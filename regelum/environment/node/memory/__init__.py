from regelum.environment.node.memory.cell import MemoryCell
from regelum.environment.node.base import Node
from typing import List, Optional, Sequence


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
