"""Input management implementations."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Optional, Tuple, Sequence

from regelum.environment.node.interfaces.base import IInputs, IResolvedInputs, IVariable
from .types import FullName


@dataclass(slots=True, frozen=True)
class Inputs(IInputs):
    """Implementation of IInputs for managing node input dependencies.

    Inputs represent the dependencies a node has on other nodes' variables.
    They can be resolved to actual variables during graph construction.
    """

    _inputs: List[FullName]  # List of full names (node_name.var_name)

    @property
    def inputs(self) -> List[FullName]:
        """Get list of input names.

        Returns:
            List of fully qualified input names.
        """
        return self._inputs

    def resolve(
        self, variables: Sequence[IVariable]
    ) -> Tuple[ResolvedInputs, Set[FullName]]:
        """Map input names to actual variables.

        Args:
            variables: List of variables to resolve against.

        Returns:
            Tuple of (resolved inputs, set of unresolved names).
        """
        var_dict = {var.full_name: var for var in variables}
        resolved = []
        unresolved = set()

        for name in self._inputs:
            if name in var_dict:
                resolved.append(var_dict[name])
            else:
                unresolved.add(name)

        return ResolvedInputs(resolved), unresolved


@dataclass(slots=True, frozen=True)
class ResolvedInputs(IResolvedInputs):
    """Implementation of IResolvedInputs for accessing resolved variables.

    ResolvedInputs provides access to the actual variables that a node
    depends on after resolution is complete.
    """

    _inputs: List[IVariable]

    def find(self, full_name: FullName) -> Optional[IVariable]:
        """Find variable by full name.

        Args:
            full_name: Fully qualified name to search for.

        Returns:
            Matching variable or None if not found.
        """
        node_name, var_name = full_name.split(".")

        for var in self._inputs:
            if var.full_name == full_name:
                return var

            if (var.name in var_name or var_name in var.name) and (
                var.node_name in node_name or node_name in var.node_name
            ):
                return var

        return None

    def __len__(self) -> int:
        """Get number of resolved inputs.

        Returns:
            Number of resolved inputs.
        """
        return len(self._inputs)

    @property
    def inputs(self) -> List[IVariable]:
        """Get list of resolved input variables.

        Returns:
            List of resolved variables.
        """
        return self._inputs
