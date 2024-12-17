from __future__ import annotations
from typing import Dict, List, Any, Tuple, Set
import numpy as np
import dask
from dask import delayed
from dask.distributed import Client, LocalCluster
import multiprocessing as mp
import os
from copy import deepcopy
from .base_new import Node, Graph
from regelum.utils.logger import logger
import time
from dask.base import visualize


def _run_node(node: Node, inputs: Dict[str, Any]) -> Tuple[Any, ...]:
    """Run a node given its input variable values."""
    # Update input variables
    if node.inputs and node.inputs.inputs:
        for inp in node.inputs.inputs:
            if inp in inputs:
                val = inputs[inp]
                var = node.resolved_inputs.find(inp)
                if var is not None:
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

    # Return results
    return tuple(var.value for var in node.variables)


class ParallelGraph(Graph):
    """A graph that executes all nodes in parallel using recursive delayed tasks."""

    def __init__(self, nodes: List[Node], debug: bool = False) -> None:
        super().__init__(nodes, debug=debug)
        self.debug = debug

        # Create Dask cluster with memory management settings
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

        # Then build dependency tree
        logger.info("\nBuilding dependency tree...")
        self.dependency_tree = self._build_dependency_tree()

    def _build_dependency_tree(self) -> Dict[Node, Set[Node]]:
        """Build a tree of node dependencies, treating graphs as atomic nodes."""
        dependencies: Dict[Node, Set[Node]] = {node: set() for node in self.nodes}

        logger.info("\n=== Building dependency tree ===")
        for node in self.nodes:
            logger.info(f"\nAnalyzing dependencies for {node.external_name}")

            # Skip if node has no inputs
            if not (node.inputs and node.inputs.inputs):
                logger.info(
                    f"  {node.external_name} has no inputs - node is independent"
                )
                continue

            # For graphs, only look at their external inputs
            node_inputs = node.inputs.inputs
            if isinstance(node, Graph):
                node_inputs = [
                    inp
                    for inp in node_inputs
                    if not any(n.external_name in inp for n in node.nodes)
                ]

            if not node_inputs:
                logger.info(f"  {node.external_name} has no external dependencies")
                continue

            logger.info(f"  External inputs for {node.external_name}: {node_inputs}")

            # Find providers for external inputs
            for input_name in node_inputs:
                provider_found = False
                for other_node in self.nodes:
                    if other_node == node:
                        continue

                    # For graph nodes, check only their outputs
                    if isinstance(other_node, Graph):
                        vars_to_check = [
                            var
                            for var in other_node.variables
                            if not any(
                                n.external_name in var.node_name
                                for n in other_node.nodes
                            )
                        ]
                    else:
                        vars_to_check = other_node.variables

                    for var in vars_to_check:
                        full_var_name = f"{other_node.external_name}.{var.name}"
                        if full_var_name == input_name:
                            if not node.is_root:
                                dependencies[node].add(other_node)
                                provider_found = True
                                logger.info(
                                    f"  {input_name} -> provided by {other_node.external_name}"
                                )
                            else:
                                logger.info(
                                    f"  {input_name} -> using previous iteration value (root node)"
                                )
                            break

                if not provider_found:
                    logger.info(f"  WARNING: No provider found for {input_name}")

        logger.info("\nFinal dependency structure:")
        for node, deps in dependencies.items():
            dep_names = [d.external_name for d in deps]
            logger.info(f"  {node.external_name} depends on: {dep_names}")

        return dependencies

    def step(self) -> None:
        """Execute one simulation step."""
        try:
            # Create initial values dict
            var_values = {
                f"{node.external_name}.{var.name}": var.value
                for node in self.nodes
                for var in node.variables
            }

            # Create tasks
            tasks = {}
            for node in self.nodes:
                # Get dependencies
                dep_inputs = {}
                for dep_node in self.dependency_tree[node]:
                    for var in dep_node.variables:
                        key = f"{dep_node.external_name}.{var.name}"
                        if node.inputs and key in node.inputs.inputs:
                            dep_inputs[key] = var_values[key]

                # Create task
                task = delayed(_run_node)(node, dep_inputs)
                tasks[node] = task

            # Compute all tasks
            futures = self.client.compute(list(tasks.values()))
            results = self.client.gather(futures)

            # Update variables
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
