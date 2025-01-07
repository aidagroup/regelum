"""Reset mechanism demonstration."""

from regelum.node.base import Node
from regelum.node.graph import Graph
from regelum.node.core.inputs import Inputs
from typing import TypeGuard, Optional


class NodeTest(Node):
    """Test node with x variable."""

    def __init__(
        self,
        name: str,
        node_input_variables: Optional[Inputs],
        node_variable_name: str,
        step_size: float,
    ) -> None:
        super().__init__(name=name, inputs=node_input_variables, step_size=step_size)
        self.x = self.define_variable(node_variable_name, value=0.0)

    def step(self) -> None:
        assert isinstance(self.x.value, float)
        self.x.value = self.x.value + 1.0


def create_complex_graph():
    # Fast updating nodes (0.01s)
    fast_node1 = NodeTest("fast1", None, "x1", 0.01)
    fast_node2 = NodeTest("fast2", Inputs(["fast1.x1"]), "x2", 0.01)
    fast_graph = Graph(
        [fast_node1, fast_node2], name="fast_graph", initialize_inner_time=True
    )

    # Medium updating nodes (0.1s)
    medium_node1 = NodeTest(
        "medium1",
        Inputs([fast_node2.get_variable("x2").full_name]),
        "x3",
        0.1,
    )
    medium_node2 = NodeTest(
        "medium2", Inputs([medium_node1.get_variable("x3").full_name]), "x4", 0.1
    )
    medium_graph = Graph(
        [medium_node1, medium_node2], name="medium_graph", initialize_inner_time=True
    )

    # Slow updating nodes (1.0s)
    slow_node1 = NodeTest(
        "slow1", Inputs([medium_node2.get_variable("x4").full_name]), "x5", 1.0
    )
    slow_node2 = NodeTest(
        "slow2", Inputs([slow_node1.get_variable("x5").full_name]), "x6", 1.0
    )

    # Create feedback loop
    feedback_node = NodeTest(
        "feedback",
        Inputs(
            [
                slow_node2.get_variable("x6").full_name,
                fast_node1.get_variable("x1").full_name,
            ]
        ),
        "x7",
        0.1,
    )

    # Main graph containing all subgraphs and nodes
    main_graph = Graph(
        [fast_graph, medium_graph, slow_node1, slow_node2, feedback_node],
        name="main_graph",
        debug=True,
        initialize_inner_time=True,
    )
    main_graph.resolve(main_graph.variables)

    return main_graph, fast_graph, medium_graph


def has_x_attribute(node: Node) -> TypeGuard[NodeTest]:
    """Type guard to check if node has x attribute."""
    return hasattr(node, "x")


if __name__ == "__main__":
    main_graph, fast_graph, medium_graph = create_complex_graph()

    print(
        f"main_graph: {[var.name for var in main_graph.variables]}\n",
        f"fast_graph: {[var.name for var in fast_graph.variables]}\n",
        f"medium_graph: {[var.name for var in medium_graph.variables]}\n",
    )

    for _ in range(20):
        main_graph.step()
        # Print values of all nodes
        print("\nStep values:")
        for node in main_graph.nodes:
            if isinstance(node, Graph):
                for inner_node in node.nodes:
                    if isinstance(inner_node, NodeTest):
                        print(f"{inner_node.external_name}: {inner_node.x.value:.3f}")
            elif has_x_attribute(node):
                print(f"{node.external_name}: {node.x.value:.3f}")

        print(f"\nTime: {main_graph.clock.time.value:.3f}")
