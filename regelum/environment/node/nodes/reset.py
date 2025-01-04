"""Reset node implementation module."""

from typing import Optional, List

from regelum.environment.node.nodes.base import Node
from regelum.environment.node.core.inputs import Inputs


class Reset(Node):
    """A node that triggers resets of other nodes."""

    def __init__(
        self, name: str = "reset", inputs: Optional[List[str] | Inputs] = None
    ) -> None:
        """Initialize Reset node."""
        super().__init__(inputs=inputs, name=name)
        self.flag = self.define_variable("flag", value=False)

    def step(self) -> None:
        """Execute reset step."""
