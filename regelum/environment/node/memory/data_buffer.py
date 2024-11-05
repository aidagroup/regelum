from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
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
        nodes_to_buffer: Union[Node, List[Node]],
        paths_to_remember: Optional[List[str]] = None,
        buffer_size: int = 1000,
        step_size: float = None,
        prefix: Optional[str] = None,
    ):
        self.nodes_to_buffer = (
            [nodes_to_buffer] if isinstance(nodes_to_buffer, Node) else nodes_to_buffer
        )
        if paths_to_remember is None:
            self.paths_to_remember = []
            for node in self.nodes_to_buffer:
                self.paths_to_remember.extend(node.state.paths)
        else:
            self.paths_to_remember = paths_to_remember

        self.buffer_size = buffer_size
        self.buffer_idx = 0
        self.is_buffer_full = False

        self.buffer_config = self._build_buffer_config()
        state = self._build_state_structure()

        super().__init__(
            step_size=step_size,
            state=state,
            inputs=self.paths_to_remember,
            prefix=prefix,
        )

    def _build_buffer_config(self) -> BufferConfig:
        leaf_names = [path.split("/")[-1] for path in self.paths_to_remember]
        shapes = []
        for path in self.paths_to_remember:
            shape = None
            for node in self.nodes_to_buffer:
                if state := node.state.search_by_path(path):
                    shape = state.shape
                    break
            if shape is None:
                raise ValueError(f"Path {path} not found in any of the provided nodes")
            shapes.append(shape)

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
            buffer_states.append(
                State(leaf_name + "_buffer", buffer_shape, np.zeros(buffer_shape))
            )

        return State("buffer", None, buffer_states)

    def compute_state_dynamics(self) -> Dict[str, Any]:
        updates = {}

        for path, leaf_name in zip(
            self.buffer_config.paths, self.buffer_config.leaf_names
        ):
            current_data = None
            for node in self.nodes_to_buffer:
                if state := node.state.search_by_path(path):
                    current_data = state.data
                    break

            buffer_data = self.state[f"buffer/{leaf_name}_buffer"].data
            buffer_data[self.buffer_idx] = current_data
            updates[f"buffer/{leaf_name}_buffer"] = buffer_data

        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        if self.buffer_idx == 0:
            self.is_buffer_full = True

        return updates

    def get_buffer_data(self) -> Dict[str, np.ndarray]:
        """Returns the valid buffer data (up to current fill level)"""
        size = self.buffer_size if self.is_buffer_full else self.buffer_idx
        return {
            leaf_name: self.state[f"buffer/{leaf_name}_buffer"].data[:size]
            for leaf_name in self.buffer_config.leaf_names
        }
