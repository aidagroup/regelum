"""Benchmark parallel execution of node graphs."""

from regelum.environment.node.nodes.base import Node
from regelum.environment.node.nodes.graph import Graph
from regelum.environment.node.core.inputs import Inputs
import numpy as np
import time
import os

# Force NumPy to use single thread
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"


def manual_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Manual matrix multiplication without BLAS."""
    n, m = a.shape
    p = b.shape[1]
    result = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            for k in range(m):
                result[i, j] += a[i, k] * b[k, j]
    return result


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
            step_size=0.1,
        )
        self.output = self.define_variable(
            "output",
            value=np.zeros(output_size),
            metadata={"shape": (output_size,)},
        )
        self.compute_time = compute_time

    def step(self) -> None:
        """Perform computation."""
        print(self.external_name)
        # CPU-heavy operations (single-threaded)
        size = 25  # Reduced size since manual computation is slower
        result = 0.0
        matrix = np.random.random((size, size))
        for _ in range(int(self.compute_time * 1e3)):
            result += np.sum(manual_matmul(matrix, matrix.T))
            matrix = 1.0 / (
                1.0 + np.exp(-matrix)
            )  # sigmoid, avoiding exp/-exp operations

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
    root = ComputeNode(inputs=[], output_size=3, compute_time=0.005, name="root")

    # First level - three branches
    node1 = ComputeNode(
        inputs=[f"{root.external_name}.output"],
        output_size=2,
        compute_time=0.2,
        name="node1",
    )

    node2 = ComputeNode(
        inputs=[f"{root.external_name}.output"],
        output_size=2,
        compute_time=0.2,
        name="node2",
    )

    node3 = ComputeNode(
        inputs=[f"{root.external_name}.output"],
        output_size=2,
        compute_time=0.005,
        name="node3",
    )

    # Second level - branch split
    node4 = ComputeNode(
        inputs=[f"{node3.external_name}.output"],
        output_size=2,
        compute_time=0.2,
        name="node4",
    )

    node5 = ComputeNode(
        inputs=[f"{node3.external_name}.output"],
        output_size=2,
        compute_time=0.2,
        name="node5",
    )

    # Merge node
    merge = ComputeNode(
        inputs=[
            f"{node1.external_name}.output",
            f"{node2.external_name}.output",
            f"{node4.external_name}.output",
            f"{node5.external_name}.output",
        ],
        output_size=1,
        compute_time=0.005,
        name="merge",
    )

    nodes = [root, node1, node2, node3, node4, node5, merge]
    return Graph(nodes, debug=False), nodes


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
    parallel_graph = Graph(nodes, debug=False).parallelize()

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
