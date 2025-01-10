"""Input management for the node system.

This module provides classes for managing node input dependencies and their resolution.
The system uses two main classes:

- Inputs: Represents unresolved dependencies as strings
- ResolvedInputs: Contains actual variable references after resolution

The resolution process maps string identifiers to actual variable instances during
graph construction.

Full names (node_name.variable_name) are crucial because:
- Multiple nodes can have variables with the same local name
- Graphs can contain multiple instances of the same node type
- Full qualification ensures correct variable resolution across the entire graph
- They enable proper dependency tracking in nested graph structures
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Optional, Tuple, Sequence

from regelum.node.interfaces.base import IInputs, IResolvedInputs, IVariable
from .types import FullName, VarName


@dataclass(slots=True, frozen=True)
class Inputs(IInputs):
    """Manages unresolved input dependencies for a node.

    Inputs track what variables a node depends on before graph construction.
    Dependencies are specified as fully qualified names (node_name.variable_name).

    Example:
        inputs = Inputs(["controller_1.action", "sensor_1.measurement"])

    Attributes:
        _inputs: List of fully qualified variable names this node depends on
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
    """Contains resolved variable references after graph construction.

    ResolvedInputs provides access to actual variable instances after the graph
    resolves dependencies. It supports fuzzy matching for variable lookup to handle
    node renaming and graph modifications.

    Example:
        resolved = ResolvedInputs([controller.action_var, sensor.measurement_var])

    Attributes:
        _inputs: List of actual variable instances this node depends on
    """

    _inputs: List[IVariable]

    def find(self, full_or_var_name: FullName | VarName) -> Optional[IVariable]:
        """Find variable by full name or variable name.

        Args:
            full_or_var_name: Fully qualified name or just variable name to search for.

        Returns:
            Matching variable or None if not found.
        """
        first_dot = full_or_var_name.find(".")
        first_bracket = full_or_var_name.find("[")

        if first_dot != -1 and (first_bracket == -1 or first_dot < first_bracket):
            node_name, var_name = full_or_var_name.split(".", 1)
            for var in self._inputs:
                if var.full_name == full_or_var_name:
                    return var

                if (var.name in var_name or var_name in var.name) and (
                    var.node_name in node_name or node_name in var.node_name
                ):
                    return var
        else:
            var_name = full_or_var_name
            for var in self._inputs:
                if var.name == var_name:
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
