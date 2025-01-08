import pytest
from typing import List, Dict, Set, Optional
import numpy as np
from unittest.mock import MagicMock

# Mock torch module
mock_torch = MagicMock()
mock_torch.float32 = "float32"
mock_torch.set_default_dtype = MagicMock()
mock_torch.Tensor = type("MockTensor", (), {})

import sys

sys.modules["torch"] = mock_torch

from regelum.node.base import Node
from regelum.node.graph import Graph
from regelum.node.core.inputs import Inputs
from regelum.node.core.variable import Variable


@pytest.fixture(autouse=True)
def reset_node_instances():
    Node._instances = {}
    yield


class DummyNode(Node):
    def __init__(
        self,
        name: str,
        step_size: Optional[float] = 0.1,
        is_continuous: bool = False,
        inputs: Optional[List[str]] = None,
    ):
        super().__init__(
            name=name,
            inputs=inputs or [],
            step_size=step_size,
            is_continuous=is_continuous,
        )
        self.step_called = 0
        self.reset_called = 0
        self.dummy_var = self.define_variable(
            "dummy_var", value=np.array([0.0]), metadata={"shape": (1,)}
        )

    def step(self) -> None:
        self.step_called += 1
        self.dummy_var.value = np.array([float(self.step_called)])

    def reset(self, *, apply_reset_modifier: bool = True) -> None:
        self.reset_called += 1
        self.dummy_var.value = np.array([0.0])
        self.step_called = 0

    def state_transition_map(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Dummy state transition for continuous nodes."""
        return np.zeros_like(state)


class DependentNode(Node):
    def __init__(
        self,
        name: str,
        dependencies: List[str],
        step_size: float = 0.1,
        is_continuous: bool = False,
    ):
        super().__init__(
            name=name,
            inputs=dependencies,
            step_size=step_size,
            is_continuous=is_continuous,
        )
        self.step_called = 0
        self.out = self.define_variable(
            "out", value=np.array([0.0]), metadata={"shape": (1,)}
        )

    def step(self) -> None:
        self.step_called += 1
        if self.resolved_inputs is not None:
            # Sum all input values
            total = 0.0
            for input_var in self.resolved_inputs.inputs:
                if input_var.value is not None:
                    total += float(input_var.value[0])
            self.out.value = np.array([total])

    def state_transition_map(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Dummy state transition for continuous nodes."""
        return np.zeros_like(state)


class ProvidingNode(Node):
    def __init__(
        self,
        name: str,
        provides: str,
        value: float = 1.0,
        step_size: float = 0.1,
        is_continuous: bool = False,
    ):
        super().__init__(
            name=name, inputs=[], step_size=step_size, is_continuous=is_continuous
        )
        self.step_called = 0
        self.provided_var = self.define_variable(
            provides, value=np.array([value]), metadata={"shape": (1,)}
        )

    def step(self) -> None:
        self.step_called += 1
        # Increment the value on each step for testing
        if self.provided_var.value is not None:
            self.provided_var.value += np.array([0.1])

    def state_transition_map(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Dummy state transition for continuous nodes."""
        return np.zeros_like(state)


class TestGraph:
    def test_init_basic(self):
        """Test basic graph initialization"""
        node1 = DummyNode("node1", step_size=0.1)
        node2 = DummyNode("node2", step_size=0.2)
        graph = Graph([node1, node2])

        assert len(graph.nodes) == 2
        assert graph.step_size == 0.1  # Should take smallest step size
        assert graph.debug is False
        assert graph.n_step_repeats == 1

    def test_step_execution(self):
        """Test that step() executes on all nodes"""
        node1 = DummyNode("node1")
        node2 = DummyNode("node2")
        graph = Graph([node1, node2])

        graph.step()
        assert node1.step_called == 1
        assert node2.step_called == 1
        assert np.allclose(node1.dummy_var.value, [1.0])
        assert np.allclose(node2.dummy_var.value, [1.0])

        graph.step()
        assert node1.step_called == 2
        assert node2.step_called == 2
        assert np.allclose(node1.dummy_var.value, [2.0])
        assert np.allclose(node2.dummy_var.value, [2.0])

    def test_reset(self):
        """Test reset functionality"""
        node1 = DummyNode("node1")
        node2 = DummyNode("node2")
        graph = Graph([node1, node2])

        # First step to change values
        graph.step()
        assert np.allclose(node1.dummy_var.value, [1.0])
        assert np.allclose(node2.dummy_var.value, [1.0])

        # Reset should set values back to 0
        graph.reset()
        assert node1.reset_called == 1
        assert node2.reset_called == 1
        assert np.allclose(node1.dummy_var.value, [0.0])
        assert np.allclose(node2.dummy_var.value, [0.0])

    def test_node_management(self):
        """Test adding and removing nodes"""
        node1 = DummyNode("node1")
        node2 = DummyNode("node2")
        graph = Graph([node1])

        assert len(graph.nodes) == 1
        graph.add_node(node2)
        assert len(graph.nodes) == 2

        graph.remove_node(node1)
        assert len(graph.nodes) == 1
        assert graph.nodes[0] == node2

    def test_step_size_validation(self):
        """Test step size validation and computation"""
        node1 = DummyNode("node1", step_size=0.1)
        node2 = DummyNode("node2", step_size=0.2)
        node3 = DummyNode("node3", step_size=0.3)

        graph = Graph([node1, node2, node3])
        assert graph.step_size == 0.1  # Should take smallest step size

    def test_continuous_discrete_mix(self):
        """Test handling of continuous and discrete nodes"""
        cont_node = DummyNode("continuous", step_size=0.1, is_continuous=True)
        disc_node = DummyNode("discrete", step_size=0.2, is_continuous=False)

        graph = Graph([cont_node, disc_node], initialize_inner_time=True)

        # Both nodes should execute
        graph.step()
        assert cont_node.step_called == 1
        assert disc_node.step_called == 1
        assert np.allclose(cont_node.dummy_var.value, [1.0])
        assert np.allclose(disc_node.dummy_var.value, [1.0])

    def test_clone_node(self):
        """Test node cloning functionality"""
        original = DummyNode("original")
        graph = Graph([original])

        cloned = graph.clone_node(original.external_name, "cloned")
        assert len(graph.nodes) == 2
        assert cloned.name == "cloned"
        assert isinstance(cloned, DummyNode)
        assert np.allclose(cloned.dummy_var.value, [0.0])

    def test_multiple_step_repeats(self):
        """Test n_step_repeats functionality"""
        node = DummyNode("node")
        graph = Graph([node], n_step_repeats=3)

        graph.step()
        assert node.step_called == 3  # Should be called 3 times per step
        assert np.allclose(node.dummy_var.value, [3.0])

    def test_hierarchical_graph_composition(self):
        """Test hierarchical composition of graphs."""
        inner_provider = ProvidingNode("inner_provider", "value")
        inner_consumer = DependentNode(
            "inner_consumer", [f"{inner_provider.external_name}.value"]
        )
        inner_graph = Graph([inner_provider, inner_consumer], name="inner")

        outer_consumer = DependentNode("outer_consumer", ["inner_consumer_1.out"])
        outer_graph = Graph([inner_graph, outer_consumer])
        outer_graph.resolve(outer_graph.variables)

        # Test value propagation through hierarchy
        outer_graph.step()
        inner_consumer_node = next(
            n for n in inner_graph.nodes if "inner_consumer" in n.external_name
        )
        outer_consumer_node = next(
            n for n in outer_graph.nodes if "outer_consumer" in n.external_name
        )
        assert np.allclose(inner_consumer_node.out.value, [1.1])
        assert np.allclose(outer_consumer_node.out.value, [1.1])

    def test_time_synchronization(self):
        """Test time synchronization between continuous and discrete nodes."""
        continuous_node = DummyNode("continuous", step_size=0.1, is_continuous=True)
        discrete_node = DummyNode("discrete", step_size=0.2, is_continuous=False)

        graph = Graph([continuous_node, discrete_node], initialize_inner_time=True)
        graph.resolve(graph.variables)

        # Run for multiple steps
        for _ in range(4):
            graph.step()

        # Discrete node should update at multiples of its step size
        assert discrete_node.step_called == 2  # Should be called twice (0.0 and 0.2)
        assert continuous_node.step_called == 4  # Should be called every 0.1

    def test_variable_mapping(self):
        """Test variable mapping and name resolution."""
        provider = ProvidingNode("provider", "value")
        middle = DependentNode("middle", [f"{provider.external_name}.value"])
        end = DependentNode("end", [f"{middle.external_name}.out"])

        graph = Graph([provider, middle, end])
        graph.resolve(graph.variables)

        # Test variable resolution through chain
        graph.step()
        assert np.allclose(provider.provided_var.value, [1.1])
        assert np.allclose(middle.out.value, [1.1])
        assert np.allclose(end.out.value, [1.1])

    def test_graph_input_resolution(self):
        """Test resolution of graph inputs."""
        external_consumer = DependentNode("consumer", ["external.value"])
        graph = Graph([external_consumer])

        inputs, unresolved = graph.resolve(graph.variables)
        assert "external.value" in inputs.inputs
        assert "external.value" in unresolved

    def test_reset_specific_nodes(self):
        """Test resetting specific nodes in graph."""
        node1 = DummyNode("node1")
        node2 = DummyNode("node2")
        graph = Graph([node1, node2])

        # Step both nodes
        graph.step()
        assert node1.step_called == 1
        assert node2.step_called == 1

        # Reset only node1
        graph.reset_nodes([node1])
        assert node1.reset_called == 1
        assert node2.reset_called == 0
        assert node1.step_called == 0
        assert node2.step_called == 1

    def test_extract_path_invalid_input(self):
        """Test error handling for invalid path extraction."""
        node1 = DummyNode("node1")
        node2 = DummyNode("node2")
        graph = Graph([node1, node2])

        with pytest.raises(ValueError, match="Path must be in format"):
            graph.extract_path_as_graph("")

        with pytest.raises(ValueError, match="Empty node names are not allowed"):
            graph.extract_path_as_graph("node1 ->  -> node2")

        with pytest.raises(ValueError, match="Could not find nodes"):
            graph.extract_path_as_graph("node1 -> nonexistent -> node2")

    def test_analyze_group_dependencies(self):
        """Test analysis of dependencies between groups."""
        # Create a chain of dependencies
        start = ProvidingNode("start", "value")
        middle1 = DependentNode("middle1", [f"{start.external_name}.value"])
        middle2 = DependentNode("middle2", [f"{middle1.external_name}.out"])
        end = DependentNode("end", [f"{middle2.external_name}.out"])

        graph = Graph([start, middle1, middle2, end])
        graph.resolve(graph.variables)

        subgraphs = graph.detect_subgraphs()
        dependencies = graph.analyze_group_dependencies(subgraphs)

        # Each group should depend on the previous one
        for i in range(1, len(subgraphs)):
            assert i - 1 in dependencies.get(i, set())

    def test_fundamental_step_size(self):
        """Test computation of fundamental step size."""
        node1 = DummyNode("node1", step_size=0.2)
        node2 = DummyNode("node2", step_size=0.3)
        node3 = DummyNode("node3", step_size=0.4)

        graph = Graph([node1, node2, node3])
        assert graph.step_size == 0.1  # GCD of step sizes

    def test_print_info(self):
        """Test graph info printing."""
        provider = ProvidingNode("provider", "value")
        consumer = DependentNode("consumer", [f"{provider.external_name}.value"])
        inner_graph = Graph([provider, consumer], name="inner")
        outer_graph = Graph([inner_graph], name="outer")

        info = outer_graph.print_info()
        assert "Graph: outer" in info
        assert "Graph: inner" in info
        assert "Node: ProvidingNode" in info
        assert "Node: DependentNode" in info

    def test_deepcopy_with_resolved_inputs(self):
        """Test deepcopy behavior with resolved inputs."""
        provider = ProvidingNode("provider", "value")
        consumer = DependentNode("consumer", [f"{provider.external_name}.value"])
        graph = Graph([provider, consumer])
        graph.resolve(graph.variables)

        import copy

        copied_graph = copy.deepcopy(graph)

        # Check that resolved inputs are properly handled
        copied_consumer = next(
            n for n in copied_graph.nodes if "consumer" in n.external_name
        )
        assert copied_consumer.resolved_inputs is None  # Should be reset on copy

        # Resolve again and check functionality
        copied_graph.resolve(copied_graph.variables)
        copied_graph.step()
        assert np.allclose(copied_consumer.out.value, [1.1])

    def test_extract_as_subgraph_error(self):
        """Test error handling in subgraph extraction."""
        node1 = DummyNode("node1")
        node2 = DummyNode("node2")
        graph = Graph([node1, node2])

        with pytest.raises(ValueError, match="Could not find nodes"):
            graph.extract_as_subgraph(["nonexistent"])

    def test_find_path(self):
        """Test path finding between nodes."""
        start = ProvidingNode("start", "value")
        middle = DependentNode("middle", [f"{start.external_name}.value"])
        end = DependentNode("end", [f"{middle.external_name}.out"])

        graph = Graph([start, middle, end])
        graph.resolve(graph.variables)

        dependencies = graph._build_dependency_graph()
        path = graph._find_path(start.external_name, end.external_name, dependencies)

        assert path == [start.external_name, middle.external_name, end.external_name]

        # Test non-existent path
        unrelated = DummyNode("unrelated")
        assert (
            graph._find_path(start.external_name, unrelated.external_name, dependencies)
            is None
        )

    def test_update_graph_node_names(self):
        """Test updating of node names in cloned graphs."""
        provider = ProvidingNode("provider", "value")
        consumer = DependentNode("consumer", [f"{provider.external_name}.value"])
        graph = Graph([provider, consumer], name="graph")
        graph.resolve(graph.variables)

        # Create a copy and update names
        import copy

        cloned = copy.deepcopy(graph)
        cloned._update_graph_node_names(cloned)

        # Check that dependencies are properly updated
        cloned_consumer = next(n for n in cloned.nodes if "consumer" in n.external_name)
        cloned_provider = next(n for n in cloned.nodes if "provider" in n.external_name)
        assert f"{cloned_provider.external_name}.value" in cloned_consumer.inputs.inputs

    def test_strongly_connected_components(self):
        """Test detection of strongly connected components."""
        node1 = DependentNode("node", ["node_2.out"])
        node2 = DependentNode("node", ["node_1.out"])
        node3 = DummyNode("node")  # Independent node

        graph = Graph([node1, node2, node3])
        graph.resolve(graph.variables)
        dependencies = graph._build_dependency_graph()

        sccs = graph._find_strongly_connected_components(dependencies)

        # Should find one SCC with node1 and node2, and one with node3
        assert set(sccs[0]) == {"node_1", "node_2"}

    def test_no_step_size_error(self):
        """Test error when no node has defined step size."""
        node1 = DummyNode("node1", step_size=None)
        node2 = DummyNode("node2", step_size=None)

        with pytest.raises(
            ValueError, match="At least one node must have a defined step_size"
        ):
            Graph([node1, node2])

    def test_clock_not_initialized_error(self):
        """Test error when accessing clock without initialization."""
        node = DummyNode("node")
        graph = Graph([node])  # No initialize_inner_time=True

        with pytest.raises(
            AttributeError, match="'Graph' object has no attribute '_clock'"
        ):
            _ = graph.clock

    def test_print_subgraphs(self):
        """Test subgraph printing functionality."""
        provider = ProvidingNode("provider", "value")
        consumer = DependentNode("consumer", [f"{provider.external_name}.value"])
        independent = DummyNode("independent")

        graph = Graph([provider, consumer, independent], debug=True)
        subgraphs = graph.detect_subgraphs()

        # Just verify it runs without error - output is for debugging
        graph.print_subgraphs(subgraphs)


class TestGraphDependencies:
    def test_dependency_resolution(self):
        """Test basic dependency resolution"""
        provider = ProvidingNode("provider", "value")
        consumer = DependentNode("consumer", [f"{provider.external_name}.value"])
        graph = Graph([provider, consumer])

        # Resolve dependencies
        graph.resolve(graph.variables)

        # Check execution order - provider should be before consumer
        assert graph.nodes.index(provider) < graph.nodes.index(consumer)

        # Test value propagation
        graph.step()
        assert np.allclose(provider.provided_var.value, [1.1])  # Initial 1.0 + 0.1
        assert np.allclose(consumer.out.value, [1.1])  # Should match provider's value

    def test_circular_dependencies(self):
        """Test detection of circular dependencies"""
        node1 = DependentNode("node1", ["node2.out"])
        node2 = DependentNode("node2", ["node1.out"])
        graph = Graph([node1, node2])

        # This should detect circular dependency
        with pytest.raises(
            ValueError, match="Couldn't resolve inputs {'node2.out'} for node node1_1"
        ):
            # First resolve to get variables
            graph.resolve(graph.variables)
            # Then try to resolve node1 which should fail due to circular dependency
            node1.resolve(list(node2.variables))

    def test_subgraph_detection(self):
        """Test detection of independent subgraphs"""
        # First subgraph
        provider1 = ProvidingNode("provider1", "value1")
        consumer1 = DependentNode("consumer1", [f"{provider1.external_name}.value1"])

        # Second independent subgraph
        provider2 = ProvidingNode("provider2", "value2")
        consumer2 = DependentNode("consumer2", [f"{provider2.external_name}.value2"])

        graph = Graph([provider1, consumer1, provider2, consumer2])
        graph.resolve(graph.variables)
        subgraphs = graph.detect_subgraphs()

        assert len(subgraphs) == 2  # Should detect 2 independent subgraphs

        # Each subgraph should have 2 nodes
        assert all(len(subgraph) == 2 for subgraph in subgraphs)

        # Check that related nodes are in the same subgraph
        for subgraph in subgraphs:
            subgraph_nodes = set(node.name for node in subgraph)
            if "provider1" in subgraph_nodes:
                assert "consumer1" in subgraph_nodes
            if "provider2" in subgraph_nodes:
                assert "consumer2" in subgraph_nodes

    def test_complex_dependency_chain(self):
        """Test resolution of complex dependency chains"""
        node1 = ProvidingNode("node1", "value1")
        node2 = DependentNode(
            "node2", [f"{node1.external_name}.value1"]
        )  # Depends on node1
        node3 = DependentNode(
            "node3", [f"{node1.external_name}.value1"]
        )  # Also depends on node1
        node4 = DependentNode(
            "node4", [f"{node2.external_name}.out", f"{node3.external_name}.out"]
        )  # Depends on both node2 and node3

        graph = Graph([node4, node2, node1, node3])  # Intentionally unordered
        graph.resolve(graph.variables)

        # Check execution order
        node_order = [node.name for node in graph.nodes]

        # node1 should be first
        assert node_order.index("node1") == 0

        # node2 and node3 should be after node1
        assert node_order.index("node2") > node_order.index("node1")
        assert node_order.index("node3") > node_order.index("node1")

        # node4 should be last
        assert node_order.index("node4") > node_order.index("node2")
        assert node_order.index("node4") > node_order.index("node3")

        # Test value propagation
        graph.step()
        assert np.allclose(node1.provided_var.value, [1.1])  # Initial 1.0 + 0.1
        assert np.allclose(node2.out.value, [1.1])  # Should match node1's value
        assert np.allclose(node3.out.value, [1.1])  # Should match node1's value
        assert np.allclose(node4.out.value, [2.2])  # Sum of node2 and node3 values

    def test_extract_subgraph(self):
        """Test extraction of subgraph with dependencies"""
        provider = ProvidingNode("provider", "value")
        middle = DependentNode("middle", [f"{provider.external_name}.value"])
        end = DependentNode("end", [f"{middle.external_name}.out"])

        graph = Graph([provider, middle, end])
        graph.resolve(graph.variables)

        # Extract subgraph starting from end node
        subgraph = graph.extract_path_as_graph(
            f"{provider.external_name} -> {middle.external_name} -> {end.external_name}"
        )

        # Should include all nodes due to dependencies
        assert len(subgraph.nodes) == 3
        assert all(
            node.name in ["provider", "middle", "end"] for node in subgraph.nodes
        )

        # Test value propagation in subgraph
        subgraph.step()
        end_node = next(node for node in subgraph.nodes if node.name == "end")
        assert np.allclose(end_node.out.value, [1.1])  # Should get the propagated value

    def test_parallel_subgraphs(self):
        """Test parallel execution paths in graph"""
        start = ProvidingNode("start", "value")

        # Path 1
        path1_1 = DependentNode("path1_1", [f"{start.external_name}.value"])
        path1_2 = DependentNode("path1_2", [f"{path1_1.external_name}.out"])

        # Path 2 (parallel to Path 1)
        path2_1 = DependentNode("path2_1", [f"{start.external_name}.value"])
        path2_2 = DependentNode("path2_2", [f"{path2_1.external_name}.out"])

        # End node depending on both paths
        end = DependentNode(
            "end", [f"{path1_2.external_name}.out", f"{path2_2.external_name}.out"]
        )

        graph = Graph([start, path1_1, path1_2, path2_1, path2_2, end])
        graph.resolve(graph.variables)

        # Test value propagation through parallel paths
        graph.step()
        assert np.allclose(start.provided_var.value, [1.1])
        assert np.allclose(path1_2.out.value, [1.1])
        assert np.allclose(path2_2.out.value, [1.1])
        assert np.allclose(end.out.value, [2.2])  # Sum of both paths
