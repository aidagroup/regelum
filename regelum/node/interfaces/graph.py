"""Interface for graph-based computation nodes that manage collections of other nodes."""

from abc import abstractmethod
from typing import (
    List,
    Dict,
    Set,
    Optional,
    Protocol,
    TypeVar,
    Generic,
    runtime_checkable,
)
from .node import INode

T = TypeVar("T", bound=INode)


@runtime_checkable
class IGraph(INode, Protocol, Generic[T]):
    """Interface for computational graphs.

    A graph is a special type of node that:
    - Contains and manages other nodes
    - Handles node dependencies and execution order
    - Manages parallel execution and subgraph detection
    - Provides graph-wide variable resolution

    Type Parameters:
        T: Type of nodes in the graph, must implement INode.
    """

    _nodes: List[T]

    @property
    @abstractmethod
    def nodes(self) -> List[T]:
        """Get list of nodes in the graph.

        Returns:
            List of all nodes contained in the graph.
        """

    @nodes.setter
    @abstractmethod
    def nodes(self, nodes: List[T]) -> None:
        """Set list of nodes in the graph."""

    @abstractmethod
    def add_node(self, node: T) -> None:
        """Add a node to the graph.

        Args:
            node: Node to add to the graph.
        """

    @abstractmethod
    def remove_node(self, node: T) -> None:
        """Remove a node from the graph.

        Args:
            node: Node to remove from the graph.
        """

    @abstractmethod
    def detect_subgraphs(self) -> List[List[T]]:
        """Detect independent subgraphs in the node network.

        Returns:
            List of node groups that can be executed independently.
        """

    @abstractmethod
    def analyze_group_dependencies(
        self, subgraphs: List[List[T]]
    ) -> Dict[int, Set[int]]:
        """Analyze dependencies between subgraph groups.

        Args:
            subgraphs: List of node groups from detect_subgraphs.

        Returns:
            Dictionary mapping group indices to their dependent group indices.
        """

    @abstractmethod
    def extract_path_as_graph(self, path: str, n_step_repeats: int = 1) -> "IGraph[T]":
        """Extract a minimal subgraph containing specified nodes.

        Args:
            path: Path specification in format 'node1 -> node2 -> node3'.
            n_step_repeats: Number of times to repeat step execution.

        Returns:
            New graph containing only the specified path.
        """

    @abstractmethod
    def extract_as_subgraph(
        self, node_names: List[str], n_step_repeats: int = 1
    ) -> "IGraph[T]":
        """Extract specified nodes into a single subgraph node.

        Args:
            node_names: List of node names to include in the subgraph
            n_step_repeats: Number of times to repeat step execution

        Returns:
            New subgraph containing the specified nodes.
        """

    @abstractmethod
    def clone_node(self, node_name: str, new_name: Optional[str] = None) -> T:
        """Clone a node and its variables.

        Args:
            node_name: Name of the node to clone.
            new_name: Optional new name for the cloned node.

        Returns:
            Cloned node instance.
        """
