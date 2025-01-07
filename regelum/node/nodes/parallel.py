"""Parallel execution of node graphs using Dask."""

from __future__ import annotations
import multiprocessing as mp
from typing import Dict, List, Any, Set, Optional, cast, TypeAlias
from dask.delayed import delayed, Delayed
from dask.distributed import Client, LocalCluster, as_completed, Future
from dask.base import visualize

from regelum.utils.logger import logger
from regelum.node.nodes.base import Node
from regelum.node.nodes.graph import Graph


NodeState = Dict[str, Any]
NodeFuture: TypeAlias = Delayed


def _extract_node_state(node: Node) -> NodeState:
    return {var.full_name: var.value for var in node.variables}


def _update_node_state(node: Node, state: NodeState) -> None:
    if isinstance(node, Graph):
        for subnode in node.nodes:
            _update_node_state(subnode, state)
    else:
        for var in node.variables:
            full_name = f"{node.external_name}.{var.name}"
            if full_name in state:
                var.value = state[full_name]


def _run_node_step(node: Node, dep_states: Dict[str, NodeState]) -> NodeState:
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
    """A graph that executes nodes in parallel using Dask."""

    def __init__(
        self,
        nodes: List[Node],
        debug: bool = False,
        n_workers: Optional[int] = None,
        threads_per_worker: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize ParallelGraph.

        Args:
            nodes: List of nodes to execute in parallel.
            debug: Whether to enable debug mode.
            n_workers: Number of workers to use.
            threads_per_worker: Number of threads per worker.
            **kwargs: Additional keyword arguments for LocalCluster.
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

    def _get_node_future(self, node: Node) -> NodeFuture:
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
        try:
            if hasattr(self, "client"):
                self.client.close()
            if hasattr(self, "cluster"):
                self.cluster.close()
        except Exception:
            pass

    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
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
            if node.resolved_inputs and node.resolved_inputs.inputs:
                for input_var in node.resolved_inputs.inputs:
                    input_name = input_var.full_name
                    if input_name in providers:
                        provider_name = providers[input_name]
                        if provider_name != node.external_name:
                            node_dependencies[node.external_name].add(provider_name)

        return node_dependencies
