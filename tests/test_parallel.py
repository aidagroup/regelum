"""Test parallel node functionality."""

import numpy as np
import pytest
from typing import Dict, Set, Optional, List
import time

from regelum.node.base import Node
from regelum.node.parallel import ParallelGraph, _extract_node_state, _update_node_state
from regelum.node.graph import Graph


@pytest.fixture
def cleanup_node_instances():
    """Clean up Node instances after each test."""
    Node._instances.clear()
    yield
    Node._instances.clear()


class ComputeNode(Node):
    """Node that performs some computation."""

    def __init__(
        self,
        name: str,
        computation_time: float = 0.1,
        inputs: Optional[List[str]] = None,
        step_size: float = 0.1,
    ):
        super().__init__(
            name=name,
            inputs=inputs or [],
            step_size=step_size,
        )
        self.computation_time = computation_time
        self.result = self.define_variable(
            "result",
            value=np.array([0.0]),
            metadata={"shape": (1,)},
        )
        self.step_count = self.define_variable(
            "step_count",
            value=np.array([0.0]),
            metadata={"shape": (1,)},
        )

    def step(self) -> None:
        time.sleep(self.computation_time)  # Simulate computation
        self.step_count.value += 1
        if self.resolved_inputs and self.resolved_inputs.inputs:
            total = 0.0
            for input_var in self.resolved_inputs.inputs:
                if input_var.value is not None:
                    total += float(input_var.value[0])
            self.result.value = np.array([total + self.step_count.value[0]])
        else:
            self.result.value = np.array([float(self.step_count.value[0])])


@pytest.mark.usefixtures("cleanup_node_instances")
class TestParallelExecution:
    """Test suite for parallel execution functionality."""

    # def test_parallel_execution_basic(self):
    #     """Test basic parallel execution of independent nodes."""
    #     nodes = [ComputeNode(f"compute_{i}", computation_time=0.5) for i in range(4)]

    #     graph = Graph(nodes)
    #     graph.resolve(variables=graph.variables)
    #     # Sequential execution time measurement
    #     start_time = time.time()

    #     graph.step()
    #     sequential_time = time.time() - start_time

    #     graph.reset()
    #     parallel_graph = graph.parallelize(n_workers=4)
    #     # Parallel execution time measurement
    #     start_time = time.time()

    #     parallel_graph.step()
    #     parallel_time = time.time() - start_time

    #     # Parallel should be faster (allowing some overhead)
    #     assert parallel_time < sequential_time

    #     # Check results are correct
    #     for node in nodes:
    #         assert np.isclose(node.result.value, [1.0])

    #     parallel_graph.close()

    def test_parallel_execution_with_dependencies(self):
        """Test parallel execution with node dependencies."""
        # Create a chain of dependent nodes
        node1 = ComputeNode("node", computation_time=0.1)
        node2 = ComputeNode(
            "node", computation_time=0.1, inputs=[f"{node1.external_name}.result"]
        )
        node3 = ComputeNode(
            "node", computation_time=0.1, inputs=[f"{node2.external_name}.result"]
        )

        graph = Graph([node1, node2, node3])
        graph.resolve(variables=graph.variables)
        parallel_graph = graph.parallelize(n_workers=3)
        parallel_graph.step()

        # Check results follow dependency chain
        assert np.isclose(node1.result.value, [1.0])  # Base computation
        assert np.isclose(node2.result.value, [2.0])  # node1's result + step_count
        assert np.isclose(node3.result.value, [3.0])  # node2's result + step_count

        parallel_graph.close()

    def test_state_extraction_and_update(self):
        """Test node state extraction and update functions."""
        node = ComputeNode("test")
        node.result.value = np.array([42.0])

        # Test state extraction
        state = _extract_node_state(node)
        print(state[f"{node.external_name}.result"])
        assert f"{node.external_name}.result" in state

        assert np.isclose(state[f"{node.external_name}.result"], np.array([42.0]))

        state[f"{node.external_name}.result"] = np.array([43.0])
        # Test state update
        _update_node_state(node, state)
        assert np.isclose(node.result.value, [43.0])

    def test_parallel_graph_with_subgraphs(self):
        """Test parallel execution with nested graphs."""
        inner_node1 = ComputeNode("inner", computation_time=0.1)
        inner_node2 = ComputeNode(
            "inner",
            computation_time=0.1,
            inputs=[f"{inner_node1.external_name}.result"],
        )
        inner_graph = Graph([inner_node1, inner_node2])
        inner_graph.resolve(variables=inner_graph.variables)

        outer_node = ComputeNode(
            "outer",
            computation_time=0.1,
            inputs=[f"{inner_node2.external_name}.result"],
        )

        graph = Graph([inner_graph, outer_node], debug=True)
        graph.resolve(variables=graph.variables)
        parallel_graph = graph.parallelize(n_workers=3)
        parallel_graph.step()

        # Check results propagate through the hierarchy
        assert np.isclose(inner_node1.result.value, [1.0])
        assert np.isclose(inner_node2.result.value, [2.0])
        assert np.isclose(outer_node.result.value, [3.0])

        parallel_graph.close()

    def test_parallel_execution_error_handling(self):
        """Test error handling in parallel execution."""

        class ErrorNode(ComputeNode):
            def step(self) -> None:
                raise ValueError("Simulated error")

        error_node = ErrorNode("error")
        normal_node = ComputeNode("normal")

        parallel_graph = ParallelGraph([error_node, normal_node], n_workers=2)

        with pytest.raises(ValueError, match="Simulated error"):
            parallel_graph.step()

        parallel_graph.close()

    def test_parallel_graph_debug_mode(self):
        """Test parallel graph execution in debug mode."""
        nodes = [ComputeNode(f"compute_{i}") for i in range(2)]

        parallel_graph = ParallelGraph(nodes, debug=True, n_workers=2)
        parallel_graph.step()

        # Debug mode should create dashboard
        assert hasattr(parallel_graph.client, "dashboard_link")

        parallel_graph.close()

    def test_worker_count_optimization(self):
        """Test automatic worker count optimization."""
        # Create more nodes than CPUs
        nodes = [ComputeNode(f"compute_{i}") for i in range(20)]

        parallel_graph = ParallelGraph(nodes)  # Should auto-adjust worker count
        parallel_graph.step()

        # Check all nodes executed correctly
        for node in nodes:
            assert np.isclose(node.result.value, [1.0])

        parallel_graph.close()

    def test_parallel_graph_multiple_steps(self):
        """Test multiple steps in parallel execution."""
        nodes = [ComputeNode(f"compute_{i}") for i in range(3)]

        parallel_graph = ParallelGraph(nodes, n_workers=3)

        # Run multiple steps
        for _ in range(3):
            parallel_graph.step()

        # Check step counts
        for node in nodes:
            assert np.isclose(node.step_count.value, [3.0])
            assert np.isclose(node.result.value, [3.0])

        parallel_graph.close()
