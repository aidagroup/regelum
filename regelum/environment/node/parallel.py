"""Parallel execution of node graphs using Dask."""

from __future__ import annotations
from typing import Dict, List, Any, Set, Optional
import numpy as np
from dask.delayed import delayed
from dask.distributed import Client, LocalCluster, as_completed
import multiprocessing as mp
from copy import deepcopy
from regelum.utils.logger import logger
from .base_new import Node, Graph
import os
import dask

NodeState = Dict[str, Any]


def _extract_node_state(node: Node) -> NodeState:
    """Extract current state from a node as a dictionary."""
    return {var.full_name: var.value for var in node.variables}


def _update_node_state(node: Node, state: NodeState) -> None:
    """Update node's state from a state dictionary."""
    for var in node.variables:
        full_name = f"{node.external_name}.{var.name}"
        if full_name in state:
            var.value = state[full_name]


def _run_node_step(
    node: Node, current_state: NodeState, dep_states: Dict[str, NodeState]
) -> NodeState:
    """Execute node step with explicit state management."""
    # Update node's state from current_state
    _update_node_state(node, current_state)

    # Update inputs from dependency states
    if node.inputs and node.inputs.inputs and node.resolved_inputs:
        for input_name in node.inputs.inputs:
            for dep_state in dep_states.values():
                if input_name in dep_state:
                    if var := node.resolved_inputs.find(input_name):
                        var.value = dep_state[input_name]
                    break

    # Execute node
    if isinstance(node, Graph):
        for _ in range(node.n_step_repeats):
            for internal_node in node.nodes:
                internal_node.step()
    else:
        node.step()

    # Return new state
    return _extract_node_state(node)


class ParallelGraph(Graph):
    def __init__(
        self,
        nodes: List[Node],
        debug: bool = False,
        n_workers: Optional[int] = None,
        threads_per_worker: int = 1,
        **kwargs,
    ):
        super().__init__(nodes, debug=debug, name="parallel_graph")
        self.debug = debug

        n_workers = n_workers or min(len(nodes), max(1, mp.cpu_count() // 2 + 1))

        # Enable diagnostics in debug mode
        if debug:
            kwargs["dashboard_address"] = ":8787"  # Enable dashboard

        self.cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            **kwargs,
        )
        self.client = Client(self.cluster)

        # Cache for futures during each step
        self._futures_cache: Dict[Node, Any] = {}

        # Build dependency tree
        self.dependency_tree = self._build_dependency_graph()

    def _get_node_future(self, node: Node) -> Any:
        """Get or create a future for a node's execution."""
        if node in self._futures_cache:
            return self._futures_cache[node]

        # Get current state
        current_state = _extract_node_state(node)

        # Get dependency futures
        dep_futures = {}
        for dep_node in self.dependency_tree[node]:
            dep_futures[dep_node.external_name] = self._get_node_future(dep_node)

        # Create node's future
        if dep_futures:
            node_future = delayed(_run_node_step, pure=False)(
                node, current_state, dep_futures
            )
        else:
            node_future = delayed(_run_node_step, pure=False)(node, current_state, {})

        self._futures_cache[node] = node_future
        return node_future

    def step(self) -> None:
        """Execute one parallel simulation step."""
        try:
            self._futures_cache.clear()

            # Create futures for all nodes
            node_futures = {node: self._get_node_future(node) for node in self.nodes}

            if self.debug:
                logger.info(f"Submitting {len(node_futures)} tasks to Dask...")

                # Visualize task graph
                dask.visualize(*node_futures.values(), filename="task_graph")
                logger.info("Task graph saved to 'task_graph.svg'")

                # Enable diagnostics dashboard
                dashboard_link = self.client.dashboard_link
                logger.info(f"Dask dashboard available at: {dashboard_link}")

            # Compute and collect results as they complete
            futures = self.client.compute(
                list(node_futures.values()), scheduler="processes"
            )

            if self.debug:
                logger.info("\nTask execution progress:")

            for future, result in as_completed(futures, with_results=True):
                idx = futures.index(future)
                print(future, "has ended")
                node = self.nodes[idx]
                _update_node_state(node, result)

                if self.debug:
                    key = future.key
                    try:
                        workers = self.client.who_has().get(future, set())
                        worker = next(iter(workers)) if workers else "cancelled"
                    except Exception:
                        worker = "unknown"
                    logger.info(
                        f"  Completed {node.external_name} on {worker} (key: {key})"
                    )

        except Exception as e:
            self.close()
            raise e

    def close(self) -> None:
        try:
            if hasattr(self, "client"):
                self.client.close()
            if hasattr(self, "cluster"):
                self.cluster.close()
        except Exception:
            pass

    def _build_dependency_graph(self) -> Dict[Node, Set[Node]]:
        """Build node dependency graph for parallel execution."""
        dependencies: Dict[Node, Set[Node]] = {node: set() for node in self.nodes}

        if self.debug:
            logger.info("\n=== Building dependency graph ===")

        # Build provider map with proper handling of nested variables
        providers = {}
        graph_nodes = [n for n in self.nodes if isinstance(n, Graph)]

        for graph_idx, node in enumerate(self.nodes, 1):
            # Map node's own variables
            for var in node.variables:
                var_name = f"{node.external_name}.{var.name}"
                providers[var_name] = node

            # If it's a graph, map all internal variables to this graph node
            if isinstance(node, Graph):
                for internal_node in node.nodes:
                    # Map both original and indexed names
                    original_name = internal_node.external_name
                    indexed_name = f"{original_name}_{graph_idx}"

                    for var in internal_node.variables:
                        # Map original internal name
                        providers[f"{original_name}.{var.name}"] = node
                        # Map indexed name
                        providers[f"{indexed_name}.{var.name}"] = node

        if self.debug:
            logger.info("\nProvider map:")
            for var_name, provider in providers.items():
                logger.info(f"  {var_name} -> {provider.external_name}")

        # Build dependencies based on inputs
        for node in self.nodes:
            if self.debug:
                logger.info(f"\nAnalyzing dependencies for {node.external_name}")

            if not (node.inputs and node.inputs.inputs):
                if self.debug:
                    logger.info(f"  {node.external_name} has no inputs")
                continue

            for input_name in node.inputs.inputs:
                if provider := providers.get(input_name):
                    if provider != node:  # Avoid self-dependencies
                        dependencies[node].add(provider)
                        if self.debug:
                            logger.info(
                                f"  {input_name} -> provided by {provider.external_name}"
                            )
                elif self.debug:
                    logger.info(f"  No provider found for {input_name}")

        if self.debug:
            logger.info("\nFinal dependency structure:")
            for node, deps in dependencies.items():
                dep_names = [d.external_name for d in deps]
                logger.info(f"  {node.external_name} depends on: {dep_names}")

        return dependencies
