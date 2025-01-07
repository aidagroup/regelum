"""Reset node implementation module."""

from typing import Optional, List

from regelum.node.nodes.base import Node
from regelum.node.core.inputs import Inputs


class Reset(Node):
    """A node that triggers resets of other nodes."""

    def __init__(
        self, name: str = "reset", inputs: Optional[List[str] | Inputs] = None
    ) -> None:
        """Initialize Reset node."""
        super().__init__(inputs=inputs, name=name, step_size=0)
        self.flag = self.define_variable("flag", value=False)

    def step(self) -> None:
        """Execute reset step."""


class ResetEachNSteps(Reset):
    """Use this node to reset a node every n steps.

    Being automatically binded to a passed node within graph.
    """

    def __init__(self, node_name_to_reset: str, n_steps: int):
        """Initialize ResetEachNSteps node.

        Args:
            node_name_to_reset: Name of the node to reset.
            n_steps: Number of steps to reset the node.
        """
        super().__init__(
            name=f"reset_{node_name_to_reset}", inputs=["step_counter_1.counter"]
        )
        self.n_steps = n_steps

    def step(self) -> None:
        self.flag.value = (
            self.resolved_inputs.find("step_counter_1.counter").value % self.n_steps
            == 0
        )
