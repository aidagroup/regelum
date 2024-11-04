from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from regelum.environment.node.base import Node, State
import numpy as np


@dataclass
class BufferConfig:
    paths: List[str]
    leaf_names: List[str]
    shape: List[Any]
    buffer_size: int


class DataBuffer(Node):
    def __init__(
        self,
        node_to_buffer: Node,
        paths_to_remember: Optional[List[str]] = None,
        buffer_size: int = 1000,
        step_size: float = None,
        prefix: Optional[str] = None,
    ):
        self.node_to_buffer = node_to_buffer
        self.paths_to_remember = paths_to_remember or [node_to_buffer.state.name]
        self.buffer_size = buffer_size
        self.buffer_idx = 0
        self.is_buffer_full = False

        self.buffer_config = self._build_buffer_config()
        state = self._build_state_structure()

        super().__init__(
            step_size=step_size,
            state=state,
            inputs=node_to_buffer.state.paths,
            prefix=prefix,
        )

    def _build_buffer_config(self) -> BufferConfig:
        leaf_names = [path.split("/")[-1] for path in self.paths_to_remember]
        shapes = [
            self.node_to_buffer.state[path].shape for path in self.paths_to_remember
        ]

        return BufferConfig(
            paths=self.paths_to_remember,
            leaf_names=leaf_names,
            shape=shapes,
            buffer_size=self.buffer_size,
        )

    def _build_state_structure(self) -> State:
        buffer_states = []
        for leaf_name, shape in zip(
            self.buffer_config.leaf_names, self.buffer_config.shape
        ):
            buffer_shape = (self.buffer_size, *shape) if shape else (self.buffer_size,)
            buffer_states.append(State(leaf_name, buffer_shape, np.zeros(buffer_shape)))

        return State("buffer", None, buffer_states)

    def compute_state_dynamics(self) -> Dict[str, Any]:
        updates = {}

        for path, leaf_name in zip(
            self.buffer_config.paths, self.buffer_config.leaf_names
        ):
            current_data = self.node_to_buffer.state[path].data
            buffer_data = self.state[f"buffer/{leaf_name}"].data
            buffer_data[self.buffer_idx] = current_data
            updates[f"buffer/{leaf_name}"] = buffer_data

        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        if self.buffer_idx == 0:
            self.is_buffer_full = True

        return updates

    def get_buffer_data(self) -> Dict[str, np.ndarray]:
        """Returns the valid buffer data (up to current fill level)"""
        size = self.buffer_size if self.is_buffer_full else self.buffer_idx
        return {
            leaf_name: self.state[f"buffer/{leaf_name}"].data[:size]
            for leaf_name in self.buffer_config.leaf_names
        }
