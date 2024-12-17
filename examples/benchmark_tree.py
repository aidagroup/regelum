"""Benchmark parallel execution of node graphs."""

from regelum.environment.node.base_new import Node, Graph, Inputs
import numpy as np
import time
import logging


class ComputeNode(Node):
    """Node that performs heavy computation."""

    def __init__(
        self,
        inputs: list[str],
        output_size: int = 1,
        compute_time: float = 0.001,
        name: str = "compute",
    ) -> None:
        super().__init__(
            inputs=inputs,
            is_continuous=False,
            is_root=not bool(inputs),
            name=name,
        )
        self.output = self.define_variable(
            "output",
            value=np.zeros(output_size),
            metadata={"shape": (output_size,)},
        )
        self.compute_time = compute_time

    def step(self) -> None:
        """Perform computation."""
        # Simulate computation
        result = 0.0
        for _ in range(int(self.compute_time * 1e6)):
            result += np.sin(np.random.random()) * np.cos(np.random.random())

        # If has inputs, combine them
        if isinstance(self.inputs, Inputs):
            value = sum(
                self.resolved_inputs.find(input_name).value
                for input_name in self.inputs.inputs
            )
        else:
            value = np.random.random(self.output.value.shape)

        self.output.value = value + result


def create_complex_graph() -> tuple[Graph, list[Node]]:
    """Create a complex computational graph with branches.

    Structure:
                   root
                    │
            ┌───────┼───────┐
            │       │       │
          node1   node2   node3
            │       │       │
            │       │    ┌──┴──┐
            │       │  node4 node5
            │       │     │    │
            └───────┼─────┼────┘
                    │
                  merge
    """
    # Root node (heavy computation)
    root = ComputeNode(inputs=[], output_size=3, compute_time=0.002, name="root")

    # First level - three branches
    node1 = ComputeNode(
        inputs=[f"{root.name}.output"], output_size=2, compute_time=0.001, name="node1"
    )

    node2 = ComputeNode(
        inputs=[f"{root.name}.output"], output_size=2, compute_time=0.002, name="node2"
    )

    node3 = ComputeNode(
        inputs=[f"{root.name}.output"], output_size=2, compute_time=0.001, name="node3"
    )

    # Second level - branch split
    node4 = ComputeNode(
        inputs=[f"{node3.name}.output"], output_size=2, compute_time=0.001, name="node4"
    )

    node5 = ComputeNode(
        inputs=[f"{node3.name}.output"], output_size=2, compute_time=0.002, name="node5"
    )

    # Merge node
    merge = ComputeNode(
        inputs=[
            f"{node1.name}.output",
            f"{node2.name}.output",
            f"{node4.name}.output",
            f"{node5.name}.output",
        ],
        output_size=1,
        compute_time=0.001,
        name="merge",
    )

    nodes = [root, node1, node2, node3, node4, node5, merge]
    return Graph(nodes), nodes


def benchmark_parallel_execution(num_steps: int = 100) -> None:
    """Benchmark sequential vs parallel execution."""
    print("\nCreating computational graph...")
    graph, nodes = create_complex_graph()
    graph.resolve(graph.variables)

    print("\nSequential Execution Analysis:")
    # Time sequential execution
    start = time.perf_counter()
    for _ in range(num_steps):
        graph.step()
    seq_time = (time.perf_counter() - start) * 1000
    print(f"Sequential step time: {seq_time:.2f}ms")

    # Reset nodes
    for node in nodes:
        node.reset()

    print("\nParallel Execution Analysis:")
    parallel_graph = Graph(nodes, debug=True).parallelize()

    # Time parallel execution
    start = time.perf_counter()
    for _ in range(num_steps):
        parallel_graph.step()
    par_time = (time.perf_counter() - start) * 1000
    print(f"Parallel step time: {par_time:.2f}ms")

    parallel_graph.close()

    # Print comparison
    print(f"\nSpeedup: {seq_time/par_time:.2f}x")
    print(f"Overhead: {par_time - seq_time:.2f}ms")

    # Verify results
    print("\nVerifying results:")
    for node in nodes:
        diff = np.abs(node.output.value).max()
        print(f"{node.name} output magnitude: {diff:.6f}")


if __name__ == "__main__":
    benchmark_parallel_execution(num_steps=10)
