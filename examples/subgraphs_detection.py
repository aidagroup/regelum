"""Benchmark parallel execution of node graphs."""

from regelum import Node, Graph, Inputs
import numpy as np
from typing import List
import time
from statistics import mean, stdev


class ComputeNode(Node):
    """Node that performs heavy computation."""

    def __init__(
        self,
        inputs: list[str],
        output_size: int = 1,
        compute_time: float = 0.001,
        name: str = None,
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
        # Simulate heavy computation with numpy operations
        size = 1000
        matrices = []
        start_time = time.perf_counter()

        for _ in range(int(self.compute_time)):
            matrix = np.random.random((size, size))
            result = np.linalg.svd(matrix)[0]
            matrices.append(result)

        if matrices:
            result = np.mean([np.sum(m) for m in matrices])
        else:
            result = 0.0

        end_time = time.perf_counter()
        # if self.debug:
        print(f"Computation time {self.external_name}: {end_time - start_time}")

        self.output.value = np.array([result])


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
    root = ComputeNode(inputs=[], output_size=2, compute_time=0.1, name="root")

    node1 = ComputeNode(
        inputs=[f"{root.external_name}.output"], output_size=2, compute_time=1.7
    )

    node2 = ComputeNode(
        inputs=[f"{root.external_name}.output"], output_size=2, compute_time=1.7
    )

    node3 = ComputeNode(
        inputs=[f"{root.external_name}.output"],
        output_size=2,
        compute_time=1.2,
    )

    node4 = ComputeNode(
        inputs=[f"{node3.external_name}.output"], output_size=2, compute_time=1.3
    )

    node5 = ComputeNode(
        inputs=[f"{node3.external_name}.output"],
        output_size=2,
        compute_time=1.3,
    )

    # Merge node
    merge = ComputeNode(
        inputs=[
            f"{node1.external_name}.output",
            f"{node2.external_name}.output",
            f"{node4.external_name}.output",
            f"{node5.external_name}.output",
        ],
        output_size=2,
        compute_time=0.1,
        name="merge",
    )

    nodes = [root, node1, node2, node3, node4, node5, merge]
    return Graph(nodes, debug=False), nodes


def print_subgraphs(graph: Graph, subgraphs: List[List[Node]]) -> None:
    """Print a text visualization of subgraphs.

    Args:
        graph: Original graph
        subgraphs: List of node groups
    """
    print("\nSubgraph Analysis")
    print("=" * 80)

    # First print group summary
    print("\nGroups Overview:")
    print("-" * 40)
    for i, group in enumerate(subgraphs):
        node_names = [node.external_name for node in group]
        print(f"Group {i}: {', '.join(node_names)}")

    # Then print detailed analysis
    print("\nDetailed Analysis:")
    print("-" * 40)
    for i, group in enumerate(subgraphs):
        print(f"\nGroup {i}:")
        print("  Nodes:")
        for node in group:
            print(f"    - {node.external_name}")
            if isinstance(node.inputs, Inputs):
                inputs_str = ", ".join(node.inputs.inputs)
                print(f"      inputs: {inputs_str}")

        # Print group dependencies
        inputs = set()
        outputs = set()
        for node in group:
            if isinstance(node.inputs, Inputs):
                inputs.update(inp.split(".")[0] for inp in node.inputs.inputs)
            for var in node.variables:
                outputs.add(f"{node.external_name}.{var.name}")

        external_inputs = {
            inp
            for inp in inputs
            if not any(any(n.name == inp for n in g_nodes) for g_nodes in subgraphs[:i])
        }

        if external_inputs:
            print("  Depends on:")
            for inp in sorted(external_inputs):
                print(f"    - {inp}")

        print("  Provides:")
        for out in sorted(outputs):
            print(f"    - {out}")

    print("\n" + "=" * 80)


def benchmark_execution(n_iterations: int = 10) -> tuple[List[float], List[float]]:
    """Benchmark parallel vs non-parallel execution.

    Args:
        n_iterations: Number of iterations for timing

    Returns:
        Tuple of (sequential_times, parallel_times)
    """
    sequential_times = []
    parallel_times = []

    for _ in range(n_iterations):
        sequential_graph, _ = create_complex_graph()
        sequential_graph.resolve(sequential_graph.variables)

        start_time = time.perf_counter()
        sequential_graph.step()
        sequential_graph.step()
        sequential_times.append(time.perf_counter() - start_time)

        sequential_graph.reset()
        parallel_graph = sequential_graph.parallelize(
            n_workers=4, processes=True, threads_per_worker=4
        )

        start_time = time.perf_counter()
        parallel_graph.step()
        parallel_graph.step()
        parallel_times.append(time.perf_counter() - start_time)

        parallel_graph.close()

    return sequential_times, parallel_times


def print_benchmark_results(
    sequential_times: List[float], parallel_times: List[float]
) -> None:
    """Print benchmark results with statistics.

    Args:
        sequential_times: List of sequential execution times
        parallel_times: List of parallel execution times
    """
    print("\nBenchmark Results")
    print("=" * 80)

    # Calculate statistics
    seq_mean = mean(sequential_times)
    seq_std = stdev(sequential_times) if len(sequential_times) > 1 else 0
    par_mean = mean(parallel_times)
    par_std = stdev(parallel_times) if len(parallel_times) > 1 else 0
    speedup = seq_mean / par_mean if par_mean > 0 else float("inf")

    # Print results
    print("\nSequential Execution:")
    print(f"  Mean time: {seq_mean:.6f} seconds")
    print(f"  Std dev:   {seq_std:.6f} seconds")
    print("\nParallel Execution:")
    print(f"  Mean time: {par_mean:.6f} seconds")
    print(f"  Std dev:   {par_std:.6f} seconds")
    print(f"\nSpeedup: {speedup:.2f}x")

    print("\nDetailed timings:")
    print("\nIteration  Sequential    Parallel    Speedup")
    print("-" * 45)
    for i, (seq_t, par_t) in enumerate(zip(sequential_times, parallel_times), 1):
        iter_speedup = seq_t / par_t if par_t > 0 else float("inf")
        print(f"{i:^9d}  {seq_t:^10.6f}  {par_t:^10.6f}  {iter_speedup:^7.2f}x")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Create and analyze graph
    # graph, nodes = create_complex_graph()
    # graph.resolve(graph.variables)
    # subgraphs = graph.detect_subgraphs()

    # # Print subgraph analysis
    # print_subgraphs(graph, subgraphs)

    # Run benchmarks
    print("\nRunning benchmarks...")
    sequential_times, parallel_times = benchmark_execution(n_iterations=1)
    print_benchmark_results(sequential_times, parallel_times)
