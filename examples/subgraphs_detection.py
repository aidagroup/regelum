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
        # Simulate computation
        result = 0.0
        for _ in range(int(self.compute_time * 1e6)):
            result += np.sin(np.random.random()) * np.cos(np.random.random())

        # If has inputs, combine them
        if isinstance(self.inputs, Inputs) and self.inputs.inputs:
            # Get first input shape as reference
            first_input = self.resolved_inputs.find(self.inputs.inputs[0]).value
            # Reshape other inputs to match first input's shape
            values = []
            for input_name in self.inputs.inputs:
                val = self.resolved_inputs.find(input_name).value
                if val.shape != first_input.shape:
                    # Take first elements if shapes don't match
                    val = val[: first_input.shape[0]]
                values.append(val)
            value = sum(values)
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
    root = ComputeNode(inputs=[], output_size=2, compute_time=0.1, name="root")

    # First level - three branches
    node1 = ComputeNode(
        inputs=[f"{root.external_name}.output"], output_size=2, compute_time=0.5
    )

    node2 = ComputeNode(
        inputs=[f"{root.external_name}.output"], output_size=2, compute_time=0.5
    )

    node3 = ComputeNode(
        inputs=[f"{root.external_name}.output"],
        output_size=2,
        compute_time=0.5,
    )

    # Second level - branch split
    node4 = ComputeNode(
        inputs=[f"{node3.external_name}.output"], output_size=2, compute_time=0.3
    )

    node5 = ComputeNode(
        inputs=[f"{node3.external_name}.output"],
        output_size=2,
        compute_time=0.3,
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
        compute_time=0.2,
        name="merge",
    )

    nodes = [root, node1, node2, node3, node4, node5, merge]
    return Graph(nodes, debug=True), nodes


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
        # Create fresh graphs for each iteration
        sequential_graph, _ = create_complex_graph()
        sequential_graph.resolve(sequential_graph.variables)

        parallel_graph, _ = create_complex_graph()
        parallel_graph.resolve(parallel_graph.variables)
        parallel_graph = parallel_graph.parallelize()

        # Time sequential execution
        start_time = time.perf_counter()
        sequential_graph.step()
        sequential_times.append(time.perf_counter() - start_time)

        # Time parallel execution
        start_time = time.perf_counter()
        parallel_graph.step()
        parallel_times.append(time.perf_counter() - start_time)

        # Clean up parallel resources
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
    print(f"\nSequential Execution:")
    print(f"  Mean time: {seq_mean:.6f} seconds")
    print(f"  Std dev:   {seq_std:.6f} seconds")
    print(f"\nParallel Execution:")
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
    graph, nodes = create_complex_graph()
    graph.resolve(graph.variables)
    subgraphs = graph.detect_subgraphs()

    # Print subgraph analysis
    print_subgraphs(graph, subgraphs)

    # Run benchmarks
    print("\nRunning benchmarks...")
    sequential_times, parallel_times = benchmark_execution(n_iterations=5)
    print_benchmark_results(sequential_times, parallel_times)
