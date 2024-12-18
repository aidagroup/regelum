"""Parallel execution of node graphs using Dask."""

from __future__ import annotations
from typing import Dict, List, Any, Tuple, Set
import numpy as np
from dask import delayed
from dask.distributed import Client, LocalCluster
import multiprocessing as mp
from copy import deepcopy
from regelum.utils.logger import logger
from .base_new import Node, Graph


def _run_node(node: Node, inputs: Dict[str, Any]) -> Tuple[Any, ...]:
    """Execute a node with given inputs and return its outputs."""
    # Update input variables
    if node.inputs and node.inputs.inputs:
        for inp in node.inputs.inputs:
            if inp in inputs:
                val = inputs[inp]
                if var := node.resolved_inputs.find(inp):
                    var.value = (
                        val.copy() if isinstance(val, np.ndarray) else deepcopy(val)
                    )

    # Execute node
    if isinstance(node, Graph):
        for _ in range(node.n_step_repeats):
            for internal_node in node.nodes:
                internal_node.step()
    else:
        node.step()

    # Return variable values
    return tuple(var.value for var in node.variables)


class ParallelGraph(Graph):
    """Graph that executes nodes in parallel using Dask."""

    def __init__(self, nodes: List[Node], debug: bool = False) -> None:
        super().__init__(nodes, debug=debug)
        self.debug = debug

        # Initialize Dask cluster
        n_workers = min(len(nodes), max(1, mp.cpu_count() // 2 + 1))
        logger.info(f"\nInitializing Dask cluster with {n_workers} workers")

        self.cluster = LocalCluster(
            n_workers=n_workers,
            processes=True,
            threads_per_worker=1,
            memory_limit="2GB",
            dashboard_address=None,
        )
        self.client = Client(self.cluster)

        # Build dependency tree
        logger.info("\nBuilding dependency tree...")
        self.dependency_tree = self._build_dependency_tree()

    def _build_dependency_tree(self) -> Dict[Node, Set[Node]]:
        """Build node dependency tree for parallel execution."""
        dependencies = {node: set() for node in self.nodes}

        if self.debug:
            logger.info("\n=== Building dependency tree ===")

        for node in self.nodes:
            if self.debug:
                logger.info(f"\nAnalyzing dependencies for {node.external_name}")

            # Skip nodes without inputs
            if not (node.inputs and node.inputs.inputs):
                if self.debug:
                    logger.info(
                        f"  {node.external_name} has no inputs - node is independent"
                    )
                continue

            # Handle graph nodes specially
            node_inputs = node.inputs.inputs
            if isinstance(node, Graph):
                node_inputs = [
                    inp
                    for inp in node_inputs
                    if not any(n.external_name in inp for n in node.nodes)
                ]

            if not node_inputs:
                if self.debug:
                    logger.info(f"  {node.external_name} has no external dependencies")
                continue

            if self.debug:
                logger.info(
                    f"  External inputs for {node.external_name}: {node_inputs}"
                )

            # Find input providers
            for input_name in node_inputs:
                provider_found = False
                for provider in self.nodes:
                    if provider == node:
                        continue

                    # Check provider's variables
                    vars_to_check = provider.variables
                    if isinstance(provider, Graph):
                        vars_to_check = [
                            var
                            for var in vars_to_check
                            if not any(
                                n.external_name in var.node_name for n in provider.nodes
                            )
                        ]

                    for var in vars_to_check:
                        if f"{provider.external_name}.{var.name}" == input_name:
                            if not node.is_root:
                                dependencies[node].add(provider)
                                provider_found = True
                                if self.debug:
                                    logger.info(
                                        f"  {input_name} -> provided by {provider.external_name}"
                                    )
                            else:
                                if self.debug:
                                    logger.info(
                                        f"  {input_name} -> using previous iteration value (root node)"
                                    )
                            break

                if not provider_found and self.debug:
                    logger.info(f"  WARNING: No provider found for {input_name}")

        if self.debug:
            logger.info("\nFinal dependency structure:")
            for node, deps in dependencies.items():
                dep_names = [d.external_name for d in deps]
                logger.info(f"  {node.external_name} depends on: {dep_names}")

        return dependencies

    def step(self) -> None:
        """Execute one parallel simulation step."""
        try:
            # Collect current variable values
            var_values = {
                f"{node.external_name}.{var.name}": var.value
                for node in self.nodes
                for var in node.variables
            }

            # Create delayed tasks
            tasks = {}
            for node in self.nodes:
                # Get required inputs
                dep_inputs = {
                    key: var_values[key]
                    for dep_node in self.dependency_tree[node]
                    for var in dep_node.variables
                    if (key := f"{dep_node.external_name}.{var.name}")
                    in node.inputs.inputs
                }

                # Create task
                tasks[node] = delayed(_run_node)(node, dep_inputs)

            # Execute tasks in parallel
            futures = self.client.compute(list(tasks.values()))
            results = self.client.gather(futures)

            # Update node variables
            for node, result in zip(tasks.keys(), results):
                for var, val in zip(node.variables, result):
                    var.value = (
                        val.copy() if isinstance(val, np.ndarray) else deepcopy(val)
                    )

        except Exception as e:
            self.close()
            raise e

    def close(self) -> None:
        """Clean up Dask resources."""
        try:
            if hasattr(self, "client"):
                self.client.close()
            if hasattr(self, "cluster"):
                self.cluster.close()
        except Exception:
            pass
