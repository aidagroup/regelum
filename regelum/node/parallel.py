"""Parallel execution of node graphs using Dask.

This module extends the Graph class to support parallel execution of nodes using Dask.
Key features:
- Task-based parallelism with dependency preservation
- Dynamic worker allocation and load balancing
- Automatic task scheduling based on input dependencies
- State synchronization between parallel tasks
- Support for both process and thread-based parallelism

The parallel execution works by:
- Converting each node's step() into a Dask task
- Preserving input dependencies between tasks
- Letting Dask scheduler handle task distribution
- Synchronizing state updates across workers
"""

from __future__ import annotations
import multiprocessing as mp
from typing import Dict, List, Any, Set, Optional, cast, TypeAlias
from dask.delayed import delayed, Delayed
from dask.distributed import Client, LocalCluster, as_completed, Future
from dask.base import visualize

from regelum.utils.logger import logger
from regelum.node.base import Node
from regelum.node.graph import Graph


NodeState = Dict[str, Any]
NodeFuture: TypeAlias = Delayed


def _extract_node_state(node: Node) -> NodeState:
    """Extract all variable values from a node for transfer between processes.

    In parallel execution, nodes run in different processes that can't directly
    share memory. We need to:
    1. Extract all variable values into a simple dictionary
    2. Make sure the data can be pickled (serialized)
    3. Preserve the full variable names for correct reassignment

    Args:
        node: Node whose state needs to be captured

    Returns:
        Dictionary mapping full variable names to their current values

    Example:
        If a node has variables 'state' and 'action', this returns:
        {
            'node_1.state': array([1.0, 2.0]),
            'node_1.action': array([0.5])
        }
    """
    return {var.full_name: var.value for var in node.variables}


def _update_node_state(node: Node, state: NodeState) -> None:
    """Update node variables with values from a state dictionary.

    When a node finishes executing on a worker process, its state needs to be
    synchronized back to the main process. This function:
    1. Takes the state dictionary from the worker
    2. Updates the corresponding variables in the main process
    3. Handles nested graphs by recursively updating their nodes

    Args:
        node: Node whose variables need updating
        state: Dictionary of variable values from worker process

    Example:
        After parallel execution, this updates the main process's node:
        _update_node_state(node, {
            'node_1.state': new_state_value,
            'node_1.action': new_action_value
        })
    """
    if isinstance(node, Graph):
        # For graph nodes, recursively update all contained nodes
        for subnode in node.nodes:
            _update_node_state(subnode, state)
    else:
        # For regular nodes, update each variable if it exists in state
        for var in node.variables:
            full_name = f"{node.external_name}.{var.name}"
            if full_name in state:
                var.value = state[full_name]


def _run_node_step(node: Node, dep_states: Dict[str, NodeState]) -> NodeState:
    """Execute a node's step on a worker process with proper state management.

    This function handles the complete lifecycle of a parallel node execution:
    1. Input Preparation:
       - Takes states from dependency nodes that finished earlier
       - Updates the node's input variables with these values

    2. Execution:
       - Runs the node's step() method in isolation
       - Ensures all required data is available locally

    3. State Capture:
       - Captures the resulting state after execution
       - Prepares it for transfer back to main process

    Args:
        node: Node to execute
        dep_states: States from dependency nodes, keyed by node name

    Returns:
        Dictionary of the node's variable values after execution

    Note:
        This function runs on worker processes, so it needs to be
        self-contained and handle all data transfer needs.
    """
    # Update input variables from dependency states
    if node.resolved_inputs and node.resolved_inputs.inputs:
        for input_var in node.resolved_inputs.inputs:
            input_name = input_var.full_name
            for dep_state in dep_states.values():
                if input_name in dep_state and (
                    var := node.resolved_inputs.find(input_name)
                ):
                    var.value = dep_state[input_name]
                    break
    for state in dep_states.values():
        _update_node_state(node, state)
    node.step()
    return _extract_node_state(node)


class ParallelGraph(Graph):
    """A graph that executes nodes in parallel using Dask.

    ParallelGraph converts node execution into a task-based parallel system where:
    - Each node's step() becomes a Dask task
    - Dependencies between nodes become task dependencies
    - Workers execute tasks as they become available
    - Results are synchronized back to the main process

    The execution process:
    1. Creates task graph based on node dependencies
    2. Submits all tasks to Dask scheduler
    3. Workers execute tasks when their dependencies complete
    4. Results are collected and synchronized

    Attributes:
        cluster: Dask LocalCluster instance
        client: Dask client for task submission
        debug: Whether to enable detailed logging
        _futures_cache: Cache of node computation futures

    Example:
        ```python
        # Create parallel graph with 4 workers
        graph = ParallelGraph(
            [node1, node2, node3],
            n_workers=4,
            threads_per_worker=1,
            debug=True  # Enables Dask dashboard
        )

        # Tasks are automatically distributed across workers
        # while preserving dependencies
        graph.step()
        ```
    """

    def __init__(
        self,
        nodes: List[Node],
        debug: bool = False,
        n_workers: Optional[int] = None,
        threads_per_worker: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize parallel execution environment.

        Sets up Dask cluster and client for distributed execution.
        Worker count defaults to CPU count if not specified.

        Args:
            nodes: List of nodes to execute in parallel
            debug: Enable detailed logging and dashboard
            n_workers: Number of worker processes
            threads_per_worker: Threads per worker process
            **kwargs: Additional arguments for LocalCluster
        """
        super().__init__(nodes, debug=debug, name="parallel_graph")
        self.debug = debug
        n_workers = n_workers or min(len(nodes), max(1, mp.cpu_count() // 2 + 1))
        if debug:
            kwargs["dashboard_address"] = ":8787"
        self.cluster = LocalCluster(
            n_workers=n_workers, threads_per_worker=threads_per_worker, **kwargs
        )
        self.client = Client(self.cluster)
        self._futures_cache: Dict[Node, NodeFuture] = {}
        self.dependency_tree = self._build_dependency_graph()
        print()

    def _get_node_future(self, node: Node) -> NodeFuture:
        """Create or retrieve a Dask task for node execution.

        Recursively builds the task graph by:
        1. Getting futures for all dependency nodes
        2. Creating a new task that depends on those futures
        3. Caching the future to avoid duplicate tasks

        Args:
            node: Node to create task for

        Returns:
            Dask Delayed object representing the node's computation

        Note:
            Tasks are created with pure=False to ensure execution
            on every step, as node state can change between steps.
        """
        if self.debug:
            print(f"Getting future for {node.external_name}")
        if node in self._futures_cache:
            return self._futures_cache[node]

        node_name = node.external_name
        dep_futures = {}
        for dep_name in self.dependency_tree[node_name]:
            dep_node = next(n for n in self.nodes if n.external_name == dep_name)
            dep_futures[dep_name] = self._get_node_future(dep_node)

        node_future = delayed(_run_node_step, pure=False)(
            node, dep_futures if dep_futures else {}
        )
        self._futures_cache[node] = node_future
        return node_future

    def _log_debug_info(self, node_futures: Dict[Node, NodeFuture]) -> None:
        logger.info(f"Submitting {len(node_futures)} tasks to Dask...")
        visualize(*node_futures.values(), filename="task_graph")
        logger.info(f"Dask dashboard available at: {self.client.dashboard_link}")

    def _process_completed_future(
        self, future: Future, node: Node, result: Any
    ) -> None:
        _update_node_state(node, result)
        if self.debug:
            who_has = self.client.who_has()
            workers = cast(Dict[Future, Set[str]], who_has).get(future, set())
            worker = next(iter(workers)) if workers else "cancelled"
            logger.info(
                f"Completed {node.external_name} on {worker} (key: {future.key})"
            )

    def step(self) -> None:
        """Execute all nodes as parallel tasks.

        The execution process:
        1. Builds task graph preserving node dependencies
        2. Submits all tasks to Dask scheduler
        3. Waits for all tasks to complete
        4. Updates node states with results

        Note:
            While tasks execute in parallel, the dependencies between
            nodes are strictly preserved - a node's task won't start
            until all its input dependencies have completed.
        """
        try:
            self._futures_cache.clear()
            node_futures = {node: self._get_node_future(node) for node in self.nodes}

            if self.debug:
                self._log_debug_info(node_futures)

            futures = list(node_futures.values())
            computed_futures = cast(
                List[Future], self.client.compute(futures, scheduler="processes")
            )
            future_to_node = dict(zip(computed_futures, node_futures))

            for future, result in as_completed(computed_futures, with_results=True):
                node = future_to_node[future]
                self._process_completed_future(future, node, result)

        except Exception as e:
            self.close()
            raise e

    def close(self) -> None:
        """Clean up Dask cluster and client resources.

        Ensures proper shutdown of:
        - Worker processes
        - Network connections
        - Dashboard if enabled
        - Temporary files and resources
        """
        try:
            if hasattr(self, "client"):
                self.client.close()
            if hasattr(self, "cluster"):
                self.cluster.close()
        except Exception:
            pass

    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """Build the task dependency graph.

        Creates a mapping of node dependencies by:
        1. Analyzing variable providers and consumers
        2. Tracking dependencies through nested graphs
        3. Handling indexed names in parallel contexts

        Returns:
            Dict mapping node names to sets of dependency node names

        Note:
            This graph is used by Dask to determine task execution order
            and optimize task distribution across workers.
        """
        node_dependencies: Dict[str, Set[str]] = {
            node.external_name: set() for node in self.nodes
        }
        providers = {}

        for graph_idx, node in enumerate(self.nodes, 1):
            for var in node.variables:
                providers[f"{node.external_name}.{var.name}"] = node.external_name

            if isinstance(node, Graph):
                for internal_node in node.nodes:
                    original_name = internal_node.external_name
                    indexed_name = f"{original_name}_{graph_idx}"
                    for var in internal_node.variables:
                        providers[f"{original_name}.{var.name}"] = node.external_name
                        providers[f"{indexed_name}.{var.name}"] = node.external_name

        for node in self.nodes:
            if (
                node.resolved_inputs
                and node.resolved_inputs.inputs
                and not node.is_root
            ):
                for input_var in node.resolved_inputs.inputs:
                    input_name = input_var.full_name
                    if input_name in providers:
                        provider_name = providers[input_name]
                        if provider_name != node.external_name:
                            node_dependencies[node.external_name].add(provider_name)

        return node_dependencies
