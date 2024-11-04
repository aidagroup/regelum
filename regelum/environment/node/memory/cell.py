from dataclasses import dataclass
from regelum.environment.node.base import Node, State
from typing import List, Optional, Dict, Any


@dataclass
class MemoryConfig:
    paths: List[str]
    leaf_names: List[str]
    shape: List[Any]
    data: List[Any]


class MemoryCell(Node):
    def __init__(
        self,
        node_to_remember: Node,
        paths_to_remember: Optional[List[str]] = None,
        step_size: float = None,
        prefix: Optional[str] = None,
    ):
        self.node_to_remember = node_to_remember
        self.paths_to_remember = paths_to_remember or [node_to_remember.state.name]

        # Initialize memory configuration
        self.memory_config = self._build_memory_config()

        # Build state structure
        state = self._build_state_structure()
        super().__init__(
            step_size=step_size,
            state=state,
            inputs=node_to_remember.state.paths,
            prefix=prefix,
        )

    def _build_memory_config(self) -> MemoryConfig:
        leaf_names = [path.split("/")[-1] for path in self.paths_to_remember]
        shapes = [
            self.node_to_remember.state[path].shape for path in self.paths_to_remember
        ]
        data = [
            self.node_to_remember.state[path].data for path in self.paths_to_remember
        ]

        return MemoryConfig(
            paths=self.paths_to_remember, leaf_names=leaf_names, shape=shapes, data=data
        )

    def _build_state_structure(self) -> State:
        def create_substates(name: str) -> State:
            return State(
                name,
                None,
                [
                    State(leaf_name, shape, data)
                    for leaf_name, shape, data in zip(
                        self.memory_config.leaf_names,
                        self.memory_config.shape,
                        self.memory_config.data,
                    )
                ],
            )

        return State(
            "memory",
            None,
            [
                self.node_to_remember.state,
                create_substates("current"),
                create_substates("previous"),
            ],
        )

    def compute_state_dynamics(self) -> Dict[str, Any]:
        updates = {}

        for path, leaf_name in zip(
            self.memory_config.paths, self.memory_config.leaf_names
        ):
            # Update previous state
            current_data = self.state[f"memory/current/{leaf_name}"].data
            updates[f"memory/previous/{leaf_name}"] = current_data

            # Update current state
            updates[f"memory/current/{leaf_name}"] = self.node_to_remember.state[
                path
            ].data

        return updates
