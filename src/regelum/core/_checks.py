from __future__ import annotations

import inspect
import sys
from collections.abc import Iterable
from fractions import Fraction
from itertools import product
from typing import Annotated, Any, cast, get_args, get_origin, get_type_hints

import z3

from regelum.core._base import (
    _MISSING,
    SYSTEM_CLOCK,
    SYSTEM_OUTPUTS,
    BaseTimeStep,
    BinaryExpr,
    BoundVarPort,
    CompileIssue,
    CompileReport,
    Connection,
    ConstExpr,
    EffectiveTransition,
    Expr,
    Guard,
    Node,
    NodeRef,
    Phase,
    PhaseRef,
    StateSnapshot,
    TerminateTarget,
    Transition,
    UnaryExpr,
    VarExpr,
    VarPort,
    VarSource,
    Z3Context,
    _class_output_paths,
    _evaluate_guard,
    _gcd_fraction,
    _is_enum_type,
    _node_input_path,
    _node_output_path,
    _normalize_output_path,
    _resolve_lazy_source,
    _source_path,
    _z3_variable_for_type,
    terminate,
)


def compile_nodes(
    nodes: Iterable[Node],
    connections: Iterable[Connection] = (),
) -> CompileReport:
    node_tuple = tuple(nodes)
    issues: list[CompileIssue] = []
    connection_map = _connection_map(connections)
    output_paths: list[str] = [
        _node_output_path(node, output)
        for node in node_tuple
        for output in node.__class__._outputs.values()
    ]
    state_vars_without_initial = tuple(
        sorted(
            _node_output_path(node, output)
            for node in node_tuple
            for output in node.__class__._outputs.values()
            if output.initial is _MISSING
        )
    )
    output_set = set(output_paths)
    class_output_paths = _class_output_paths(node_tuple)
    inputs: dict[str, str] = {}

    duplicate_state_vars = {path for path in output_set if output_paths.count(path) > 1}
    for path in sorted(duplicate_state_vars):
        issues.append(
            CompileIssue(
                location=path,
                message="state path is declared more than once",
            )
        )

    node_ids = [node.node_id for node in node_tuple]
    for node_id in sorted({node_id for node_id in set(node_ids) if node_ids.count(node_id) > 1}):
        issues.append(
            CompileIssue(
                location=node_id,
                message="node name is declared more than once",
            )
        )

    for node in node_tuple:
        if node.__class__._input_declaration_error is not None:
            issues.append(
                CompileIssue(
                    location=f"{node.__class__.__name__}.update",
                    message=node.__class__._input_declaration_error,
                )
            )
        for output in node.__class__._outputs.values():
            if output.initial is _MISSING:
                continue
            try:
                output.initial_value(node)
            except Exception as exc:
                issues.append(
                    CompileIssue(
                        location=_node_output_path(node, output),
                        message=f"state variable initial value failed: {exc}",
                    )
                )

    for node in node_tuple:
        for name, input_port in node.__class__._inputs.items():
            location = _node_input_path(node, input_port)
            source_ref = connection_map.get(location, input_port.source)
            if source_ref is None:
                issues.append(
                    CompileIssue(
                        location=location,
                        message="input source is not connected",
                    )
                )
                continue
            try:
                resolved_source_ref = _resolve_lazy_source(source_ref)
                source = _source_path(source_ref)
            except Exception as exc:
                issues.append(
                    CompileIssue(
                        location=location,
                        message=f"cannot resolve input source: {exc}",
                    )
                )
                continue
            inputs[location] = source
            candidates = class_output_paths.get(source, ())
            is_class_level_output_ref = isinstance(resolved_source_ref, VarPort)
            if len(candidates) > 1 and (is_class_level_output_ref or source not in output_set):
                issues.append(
                    CompileIssue(
                        location=location,
                        message=(
                            f"ambiguous input source {source!r}; "
                            f"candidates are {tuple(candidates)}; "
                            "use instance connection via port(...).connect(...)"
                        ),
                    )
                )
            elif source not in output_set and source not in SYSTEM_OUTPUTS:
                issues.append(
                    CompileIssue(
                        location=location,
                        message=f"unknown input source {source!r}",
                    )
                )

    return CompileReport(
        nodes=tuple(node.node_id for node in node_tuple),
        inputs=inputs,
        state_vars=tuple(sorted(output_set)),
        issues=tuple(issues),
        warnings=(),
        state_vars_without_initial=state_vars_without_initial,
    )


def _required_initial_issues(
    required_initial_state_vars: dict[str, tuple[str, ...]],
) -> list[CompileIssue]:
    return [
        CompileIssue(
            location=source_path,
            message=(f"state variable initial value is required before first read by {readers}"),
        )
        for source_path, readers in required_initial_state_vars.items()
    ]


def _check_phase_graph_completeness(
    phases: tuple[Phase, ...],
    phase_nodes: tuple[Node, ...],
    connection_map: dict[str, VarSource[Any]],
) -> list[CompileIssue]:
    covered_node_ids = {node.node_id for node in phase_nodes}
    issues: list[CompileIssue] = []

    for node in phase_nodes:
        for input_port in node.__class__._inputs.values():
            location = _node_input_path(node, input_port)
            source_ref = connection_map.get(location, input_port.source)
            if source_ref is None:
                continue
            _append_missing_bound_source_issue(
                issues,
                location,
                source_ref,
                covered_node_ids,
            )

    for phase in phases:
        for transition in phase.transitions:
            for source_ref in _guard_sources(transition.predicate):
                _append_missing_bound_source_issue(
                    issues,
                    f"{phase.name}.{transition.name}",
                    source_ref,
                    covered_node_ids,
                )
    return issues


def _append_missing_bound_source_issue(
    issues: list[CompileIssue],
    location: str,
    source_ref: VarSource[Any],
    covered_node_ids: set[str],
) -> None:
    try:
        resolved_source_ref = _resolve_lazy_source(source_ref)
    except Exception:
        return
    if not isinstance(resolved_source_ref, BoundVarPort):
        return
    source_node = resolved_source_ref.node
    if source_node.node_id in covered_node_ids:
        return
    issues.append(
        CompileIssue(
            location=location,
            message=(
                "incomplete phase graph: "
                f"{location} reads {resolved_source_ref.path}, but node "
                f"{source_node.node_id} is not assigned to any phase"
            ),
        )
    )


def _guard_sources(guard: Guard) -> tuple[VarSource[Any], ...]:
    if isinstance(guard, VarExpr):
        var_source = cast(VarSource[Any] | None, guard.__dict__.get("var_source"))
        if var_source is not None:
            return (var_source,)
    if isinstance(guard, UnaryExpr):
        return _guard_sources(guard.operand)
    if isinstance(guard, BinaryExpr):
        return (*_guard_sources(guard.left), *_guard_sources(guard.right))
    return ()


def _check_transition_structure(phase: Phase) -> tuple[list[CompileIssue], list[CompileIssue]]:
    issues: list[CompileIssue] = []
    warnings: list[CompileIssue] = []
    transitions = phase.transitions
    if not transitions:
        return issues, warnings

    goto_transitions = [transition for transition in transitions if transition.kind == "goto"]
    if goto_transitions:
        if len(transitions) != 1:
            issues.append(
                CompileIssue(
                    location=phase.name,
                    message="Goto transitions cannot be mixed with If/ElseIf/Else chains",
                )
            )
        return issues, warnings

    has_open_chain = False
    chain_closed = False
    last_else_name: str | None = None
    for transition in transitions:
        if last_else_name is not None and transition.kind == "if":
            warnings.append(
                CompileIssue(
                    location=f"{phase.name}.{transition.name}",
                    message=(
                        f"transition follows Else {last_else_name!r}; "
                        "move it before Else or split the control flow into another phase"
                    ),
                )
            )
        if transition.kind == "if":
            has_open_chain = True
            chain_closed = False
            last_else_name = None
        elif transition.kind == "elseif":
            if not has_open_chain:
                issues.append(
                    CompileIssue(
                        location=f"{phase.name}.{transition.name}",
                        message="ElseIf must follow If or ElseIf",
                    )
                )
            elif chain_closed:
                issues.append(
                    CompileIssue(
                        location=f"{phase.name}.{transition.name}",
                        message="ElseIf must follow If or ElseIf",
                    )
                )
        elif transition.kind == "else":
            if not has_open_chain:
                issues.append(
                    CompileIssue(
                        location=f"{phase.name}.{transition.name}",
                        message="Else must follow If or ElseIf",
                    )
                )
            elif chain_closed:
                issues.append(
                    CompileIssue(
                        location=f"{phase.name}.{transition.name}",
                        message="Else must follow If or ElseIf",
                    )
                )
            chain_closed = True
            last_else_name = transition.name
    return issues, warnings


def _effective_transitions(phase: Phase) -> tuple[EffectiveTransition, ...]:
    return _effective_transition_list(phase.transitions)


def _effective_transition_list(
    transitions: tuple[Transition, ...],
) -> tuple[EffectiveTransition, ...]:
    if len(transitions) == 1:
        transition = transitions[0]
        if transition.kind == "goto":
            return (
                EffectiveTransition(
                    predicate=ConstExpr(True),
                    target=transition.target,
                    name=transition.name,
                ),
            )

    effective: list[EffectiveTransition] = []
    previous_in_chain: list[Guard] = []
    chain_closed = False
    has_open_chain = False
    for transition in transitions:
        if transition.kind == "goto":
            return (
                EffectiveTransition(
                    predicate=ConstExpr(True),
                    target=transition.target,
                    name=transition.name,
                ),
            )
        if transition.kind == "if":
            previous_in_chain = []
            chain_closed = False
            has_open_chain = True
            predicate = transition.predicate
            previous_in_chain.append(transition.predicate)
        elif transition.kind == "elseif" and has_open_chain and not chain_closed:
            predicate = _guard_after_previous_failed(
                transition.predicate,
                tuple(previous_in_chain),
            )
            previous_in_chain.append(transition.predicate)
        elif transition.kind == "else" and has_open_chain and not chain_closed:
            predicate = _all_previous_guards_failed(tuple(previous_in_chain))
            chain_closed = True
        else:
            predicate = transition.predicate

        effective.append(
            EffectiveTransition(
                predicate=predicate,
                target=transition.target,
                name=transition.name,
            )
        )
    return tuple(effective)


def _guard_after_previous_failed(predicate: Guard, previous: tuple[Guard, ...]) -> Guard:
    if isinstance(predicate, Expr) and all(isinstance(guard, Expr) for guard in previous):
        return _all_previous_exprs_failed(cast(tuple[Expr, ...], previous)) & predicate

    def guard(state: StateSnapshot) -> bool:
        return all(
            not _evaluate_guard(previous_guard, state) for previous_guard in previous
        ) and _evaluate_guard(predicate, state)

    return guard


def _all_previous_guards_failed(previous: tuple[Guard, ...]) -> Guard:
    if all(isinstance(guard, Expr) for guard in previous):
        return _all_previous_exprs_failed(cast(tuple[Expr, ...], previous))

    def guard(state: StateSnapshot) -> bool:
        return all(not _evaluate_guard(previous_guard, state) for previous_guard in previous)

    return guard


def _all_previous_exprs_failed(previous: tuple[Expr, ...]) -> Expr:
    if not previous:
        return ConstExpr(True)
    expr = previous[0]
    for guard in previous[1:]:
        expr = expr | guard
    return ~expr


def _check_phase_guard_references(
    phase: Phase,
    outputs_by_path: dict[str, str],
    class_output_paths: dict[str, tuple[str, ...]],
) -> list[CompileIssue]:
    issues: list[CompileIssue] = []
    for transition in phase.transitions:
        if not isinstance(transition.predicate, Expr):
            continue
        for variable in sorted(transition.predicate.variables):
            try:
                _resolve_guard_variable_path(
                    variable,
                    outputs_by_path,
                    class_output_paths,
                )
            except ValueError as exc:
                issues.append(
                    CompileIssue(
                        location=f"{phase.name}.{transition.name}",
                        message=str(exc),
                    )
                )
                continue
    return issues


def _resolve_guard_variable_path(
    variable: str,
    outputs_by_path: dict[str, str],
    class_output_paths: dict[str, tuple[str, ...]],
) -> str:
    variable = _normalize_output_path(variable)
    if variable in SYSTEM_OUTPUTS:
        return variable
    try:
        return outputs_by_path[variable]
    except KeyError:
        candidates = class_output_paths.get(variable, ())
        if len(candidates) > 1:
            raise ValueError(
                f"ambiguous guard variable {variable!r}; "
                f"candidates are {tuple(candidates)}; use instance state reference"
            )
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(f"unknown guard variable {variable!r}")


def _run_requires_inputs(node: Node) -> bool:
    try:
        signature = inspect.signature(node.update)
    except (TypeError, ValueError):
        return True
    state_parameter_names = set(node.__class__._update_state_parameter_names)
    return any(
        name not in state_parameter_names
        and parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        for name, parameter in signature.parameters.items()
    )


def _connection_map(
    connections: Iterable[Connection],
) -> dict[str, VarSource[Any]]:
    return {connection.input.path: connection.source for connection in connections}


def _node_connections(nodes: Iterable[Node]) -> tuple[Connection, ...]:
    return tuple(connection for node in nodes for connection in node._connections.values())


def _nodes_from_phases(phases: tuple[Phase, ...]) -> tuple[Node, ...]:
    nodes: list[Node] = []
    seen: set[int] = set()
    for phase in phases:
        for node in phase.nodes:
            for expanded_node in _expand_phase_node(node):
                identity = id(expanded_node)
                if identity in seen:
                    continue
                nodes.append(expanded_node)
                seen.add(identity)
    return tuple(nodes)


def _expand_phase_node(node: Node) -> tuple[Node, ...]:
    if _is_ode_system(node):
        return (node, *_ode_system_nodes(node))
    return (node,)


def _is_ode_system(node: Node) -> bool:
    return bool(getattr(node, "_is_ode_system", False))


def _ode_system_dt(node: Node) -> Fraction:
    return cast(Any, node).dt


def _ode_system_nodes(node: Node) -> tuple[Any, ...]:
    return tuple(cast(Any, node).nodes)


def _is_continuous_phase(phase: Phase) -> bool:
    return bool(phase.nodes) and all(_is_ode_system(node) for node in phase.nodes)


def _is_discrete_phase(phase: Phase) -> bool:
    return all(not _is_ode_system(node) for node in phase.nodes)


def _check_phase_kinds(phases: tuple[Phase, ...]) -> list[CompileIssue]:
    issues: list[CompileIssue] = []
    for phase in phases:
        if _is_continuous_phase(phase) or _is_discrete_phase(phase):
            continue
        issues.append(
            CompileIssue(
                location=phase.name,
                message="phase cannot mix ODESystem nodes with ordinary nodes",
            )
        )
    return issues


def _check_cross_ode_system_coupling(
    phases: tuple[Phase, ...],
    inputs: dict[str, str],
) -> list[CompileIssue]:
    issues: list[CompileIssue] = []
    for phase in phases:
        if not _is_continuous_phase(phase) or len(phase.nodes) < 2:
            continue
        ode_system_by_node_id: dict[str, str] = {}
        for ode_system in phase.nodes:
            for ode_node in _ode_system_nodes(ode_system):
                ode_system_by_node_id[ode_node.node_id] = ode_system.node_id
        for ode_system in phase.nodes:
            ode_system_id = ode_system.node_id
            for ode_node in _ode_system_nodes(ode_system):
                for input_port in ode_node.__class__._inputs.values():
                    input_path = _node_input_path(ode_node, input_port)
                    source_path = inputs.get(input_path)
                    if source_path is None:
                        continue
                    source_node_id = source_path.split(".", maxsplit=1)[0]
                    source_system_id = ode_system_by_node_id.get(source_node_id)
                    if source_system_id is None or source_system_id == ode_system_id:
                        continue
                    issues.append(
                        CompileIssue(
                            location=input_path,
                            message=(
                                "continuous phase contains coupled ODESystem nodes: "
                                f"{input_path} reads {source_path} across ODESystem "
                                f"boundary {source_system_id!r} -> {ode_system_id!r}. "
                                "Put continuously coupled ODENodes into the same ODESystem."
                            ),
                        )
                    )
    return issues


def _check_continuous_phase_contract(
    phases: tuple[Phase, ...],
    nodes: tuple[Node, ...],
    *,
    initial_phase: str,
    max_phase_steps: int,
) -> list[CompileIssue]:
    continuous_phases = {phase.name for phase in phases if _is_continuous_phase(phase)}
    if not continuous_phases:
        return []
    if len(continuous_phases) != 1:
        return []
    phase_map = {phase.name: phase for phase in phases}
    if initial_phase not in phase_map:
        return []

    output_types = _output_types(nodes)
    output_types[f"{SYSTEM_CLOCK}.tick"] = int
    output_types[f"{SYSTEM_CLOCK}.time"] = float
    domains = _finite_domains_by_path(nodes)
    guard_vars = _phase_guard_variables(phases)
    initial_bindings: dict[str, Any] = {}
    initial_constraints: list[z3.BoolRef] = []

    issues: list[CompileIssue] = []
    stack: list[tuple[str, int, int, dict[str, Any], list[z3.BoolRef], tuple[str, ...]]] = [
        (initial_phase, 0, 0, initial_bindings, initial_constraints, ())
    ]
    while stack:
        phase_name, depth, continuous_count, bindings, constraints, path = stack.pop()
        if depth >= max_phase_steps:
            continue
        phase = phase_map[phase_name]
        next_count = continuous_count + int(phase.name in continuous_phases)
        next_path = (*path, phase.name)
        if next_count > 1:
            issue = _continuous_contract_issue(
                "path can reach a continuous phase more than once",
                next_path,
                output_types,
                domains,
                bindings,
                constraints,
            )
            if issue is not None:
                issues.append(issue)
                return issues
            continue

        phase_bindings = dict(bindings)
        phase_constraints = list(constraints)
        try:
            _havoc_phase_writes(
                phase,
                depth=depth,
                relevant_vars=guard_vars,
                output_types=output_types,
                domains=domains,
                bindings=phase_bindings,
                constraints=phase_constraints,
            )
        except Exception as exc:
            return [
                CompileIssue(
                    location=phase.name,
                    message=f"continuous phase contract check failed: {exc}",
                )
            ]

        for transition in _effective_transitions(phase):
            if not isinstance(transition.predicate, Expr):
                return [
                    CompileIssue(
                        location=f"{phase.name}.{transition.name}",
                        message=(
                            "continuous phase contract cannot be proven with "
                            "non-symbolic transition guards"
                        ),
                    )
                ]
            target = _phase_ref_name(transition.target)
            transition_bindings = dict(phase_bindings)
            transition_constraints = list(phase_constraints)
            try:
                predicate = _z3_guard_for_transition(
                    transition,
                    output_types=output_types,
                    domains=domains,
                    bindings=transition_bindings,
                    constraints=transition_constraints,
                )
            except Exception as exc:
                return [
                    CompileIssue(
                        location=f"{phase.name}.{transition.name}",
                        message=f"continuous phase contract check failed: {exc}",
                    )
                ]
            transition_constraints.append(predicate)
            if not _z3_constraints_satisfiable(transition_constraints):
                continue
            if target is None:
                if next_count != 1:
                    issue = _continuous_contract_issue(
                        "path can terminate without reaching a continuous phase",
                        next_path,
                        output_types,
                        domains,
                        transition_bindings,
                        transition_constraints,
                    )
                    if issue is not None:
                        issues.append(issue)
                        return issues
                continue
            if target in phase_map:
                stack.append(
                    (
                        target,
                        depth + 1,
                        next_count,
                        transition_bindings,
                        transition_constraints,
                        next_path,
                    )
                )
    return issues


def _havoc_phase_writes(
    phase: Phase,
    *,
    depth: int,
    relevant_vars: frozenset[str],
    output_types: dict[str, type[Any]],
    domains: dict[str, tuple[Any, ...]],
    bindings: dict[str, Any],
    constraints: list[z3.BoolRef],
) -> None:
    for path, _port in _phase_state_writes(phase):
        if path not in relevant_vars:
            continue
        output_type = output_types.get(path)
        if output_type is None:
            continue
        bindings[path] = _z3_variable_for_type(output_type, f"phase::{depth}::{phase.name}::{path}")
        ctx = Z3Context(output_types, domains=domains, bindings=bindings)
        constraints.extend(ctx.domain_constraints((path,)))


def _phase_state_writes(phase: Phase) -> Iterable[tuple[str, VarPort[Any]]]:
    for node in phase.nodes:
        for expanded_node in _expand_phase_node(node):
            yield from _node_ref_outputs(expanded_node)


def _phase_guard_variables(phases: tuple[Phase, ...]) -> frozenset[str]:
    variables: set[str] = set()
    for phase in phases:
        for transition in _effective_transitions(phase):
            if isinstance(transition.predicate, Expr):
                variables.update(transition.predicate.variables)
    return frozenset(variables)


def _z3_guard_for_transition(
    transition: EffectiveTransition,
    *,
    output_types: dict[str, type[Any]],
    domains: dict[str, tuple[Any, ...]],
    bindings: dict[str, Any],
    constraints: list[z3.BoolRef],
) -> z3.BoolRef:
    predicate = cast(Expr, transition.predicate)
    ctx = Z3Context(output_types, domains=domains, bindings=bindings)
    result = predicate.to_z3(ctx)
    bindings.update(ctx.variables)
    constraints.extend(ctx.domain_constraints(predicate.variables))
    if z3.is_bool(result):
        return result
    raise TypeError("transition guard did not compile to a z3 Bool expression")


def _z3_constraints_satisfiable(constraints: list[z3.BoolRef]) -> bool:
    solver = z3.Solver()
    solver.add(*constraints)
    return solver.check() == z3.sat


def _continuous_contract_issue(
    message: str,
    path: tuple[str, ...],
    output_types: dict[str, type[Any]],
    domains: dict[str, tuple[Any, ...]],
    bindings: dict[str, Any],
    constraints: list[z3.BoolRef],
) -> CompileIssue | None:
    solver = z3.Solver()
    solver.add(*constraints)
    if solver.check() != z3.sat:
        return None
    ctx = Z3Context(output_types, domains=domains, bindings=bindings)
    return CompileIssue(
        location=" -> ".join(path),
        message=(
            f"continuous phase contract violation: {message}; "
            f"witness={_model_snapshot(solver.model(), ctx, limit=16)}"
        ),
    )


def _check_clock_name_is_reserved(nodes: tuple[Node, ...]) -> list[CompileIssue]:
    return [
        CompileIssue(
            location=node.node_id,
            message="Clock is a reserved system source name",
        )
        for node in nodes
        if node.node_id == SYSTEM_CLOCK
    ]


def _schedule_warnings(
    *,
    base_dt: Fraction,
    requested_base_dt: BaseTimeStep,
    continuous: bool,
    explicit_dts: tuple[Fraction, ...],
) -> list[CompileIssue]:
    if continuous or not explicit_dts:
        return []
    common = _gcd_fraction(explicit_dts)
    if common <= base_dt:
        return []
    if requested_base_dt == "auto":
        message = (
            f'discrete-only system uses base_dt=1 for base_dt="auto", but all explicit '
            f"node dt values are multiples of {common}; this creates idle ticks. "
            f"Set base_dt={common!s} to compress the schedule."
        )
    else:
        message = (
            f"explicit base_dt={base_dt} creates idle ticks because all explicit node dt "
            f"values are multiples of {common}. Set base_dt={common!s} if this is not "
            "intentional."
        )
    return [CompileIssue(location="base_dt", message=message)]


def _phase_ref_name(phase_ref: PhaseRef) -> str | None:
    if phase_ref is None or phase_ref is terminate:
        return None
    if isinstance(phase_ref, Phase):
        return phase_ref.name
    if isinstance(phase_ref, TerminateTarget):
        return None
    return phase_ref


def _phase_dependency_edges(
    phase_node_ids: tuple[str, ...],
    inputs: dict[str, str],
) -> list[tuple[str, str]]:
    active = set(phase_node_ids)
    edges: set[tuple[str, str]] = set()
    for target_node in active:
        prefix = f"{target_node}."
        for input_path, source_path in inputs.items():
            if not input_path.startswith(prefix):
                continue
            source_node = source_path.split(".", maxsplit=1)[0]
            if source_node in active and source_node != target_node:
                edges.add((source_node, target_node))
    return sorted(edges)


def _node_ref_outputs(node_ref: NodeRef) -> Iterable[tuple[str, VarPort[Any]]]:
    for output in node_ref.__class__._outputs.values():
        yield _node_output_path(node_ref, output), output


def _has_cycle(nodes: list[str], edges: list[tuple[str, str]]) -> bool:
    return _topological_order(tuple(nodes), edges) is None


def _topological_order(
    nodes: tuple[str, ...],
    edges: list[tuple[str, str]],
) -> tuple[str, ...] | None:
    node_set = set(nodes)
    order_index = {node: index for index, node in enumerate(nodes)}
    adjacency: dict[str, list[str]] = {node: [] for node in nodes}
    indegree: dict[str, int] = {node: 0 for node in nodes}
    for source, target in edges:
        if source not in node_set or target not in node_set:
            continue
        adjacency.setdefault(source, []).append(target)
        indegree[target] = indegree.get(target, 0) + 1
    for targets in adjacency.values():
        targets.sort(key=order_index.__getitem__)

    queue = [node for node in nodes if indegree.get(node, 0) == 0]
    schedule: list[str] = []
    while queue:
        node = queue.pop(0)
        schedule.append(node)
        for target in adjacency.get(node, []):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
                queue.sort(key=order_index.__getitem__)
    if len(schedule) != len(nodes):
        return None
    return tuple(schedule)


def _check_c2star(
    phases: tuple[Phase, ...],
    nodes: tuple[Node, ...],
    *,
    depth: int | None,
    max_depth: int,
) -> list[CompileIssue]:
    """Certify bounded residence in every cyclic phase SCC under F_max.

    Only UNSAT certifies termination. SAT describes an arbitrary SCC entry,
    not necessarily a reachable tick state; UNKNOWN is always inconclusive.
    Depths count transitions, not cycle traversals.
    """
    for name, value in (("depth", depth), ("max_depth", max_depth)):
        if value is None and name == "depth":
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"C2* {name} must be a positive integer")
    phase_map = {phase.name: phase for phase in phases}
    output_types = _output_types(nodes)
    domains = _finite_domains_by_path(nodes)
    issues: list[CompileIssue] = []
    for component in _cyclic_sccs(list(phase_map), _phase_transition_edges(phases)):
        try:
            issue = _check_scc_paths(
                component,
                phase_map,
                output_types,
                domains,
                depth=depth,
                max_depth=max_depth,
            )
        except (KeyError, TypeError, ValueError, z3.Z3Exception) as exc:
            issue = CompileIssue(
                location="SCC {" + ", ".join(component) + "}",
                message=f"C2* inconclusive: unsupported symbolic encoding ({exc})",
            )
        if issue is not None:
            issues.append(issue)
    return issues


def _phase_transition_edges(phases: tuple[Phase, ...]) -> list[tuple[str, str]]:
    return sorted(
        (phase.name, target)
        for phase in phases
        for transition in _effective_transitions(phase)
        for target in (_phase_ref_name(transition.target),)
        if target is not None
    )


def _check_phase_reachability(
    phases: tuple[Phase, ...],
    initial_phase: str,
) -> list[CompileIssue]:
    phase_names = {phase.name for phase in phases}
    if initial_phase not in phase_names:
        return []
    adjacency: dict[str, list[str]] = {phase.name: [] for phase in phases}
    for source, target in _phase_transition_edges(phases):
        if target in phase_names:
            adjacency[source].append(target)

    reachable: set[str] = set()
    stack = [initial_phase]
    while stack:
        phase_name = stack.pop()
        if phase_name in reachable:
            continue
        reachable.add(phase_name)
        stack.extend(adjacency.get(phase_name, ()))

    return [
        CompileIssue(
            location=phase.name,
            message=f"phase is unreachable from initial phase {initial_phase!r}",
        )
        for phase in phases
        if phase.name not in reachable
    ]


def _cyclic_sccs(
    nodes: list[str],
    edges: list[tuple[str, str]],
) -> tuple[tuple[str, ...], ...]:
    """Iterative Kosaraju traversal; include singleton self-loops only."""
    adjacency: dict[str, list[str]] = {node: [] for node in nodes}
    reverse: dict[str, list[str]] = {node: [] for node in nodes}
    edge_set = set(edges)
    for source, target in edges:
        if source in adjacency and target in adjacency:
            adjacency[source].append(target)
            reverse[target].append(source)
    seen: set[str] = set()
    order: list[str] = []
    for root in nodes:
        if root in seen:
            continue
        seen.add(root)
        stack = [(root, iter(adjacency[root]))]
        while stack:
            node, targets = stack[-1]
            target = next(targets, None)
            if target is None:
                order.append(node)
                stack.pop()
            elif target not in seen:
                seen.add(target)
                stack.append((target, iter(adjacency[target])))
    seen.clear()
    components = []
    for root in reversed(order):
        if root in seen:
            continue
        seen.add(root)
        pending = [root]
        component = []
        while pending:
            node = pending.pop()
            component.append(node)
            for target in reverse[node]:
                if target not in seen:
                    seen.add(target)
                    pending.append(target)
        if len(component) > 1 or (root, root) in edge_set:
            components.append(tuple(sorted(component)))
    return tuple(sorted(components))


def _check_scc_paths(
    component: tuple[str, ...],
    phases: dict[str, Phase],
    output_types: dict[str, type[Any]],
    domains: dict[str, tuple[Any, ...]],
    *,
    depth: int | None,
    max_depth: int,
) -> CompileIssue | None:
    location = "SCC {" + ", ".join(component) + "}"
    internal = [
        (source, _phase_ref_name(t.target), t.predicate)
        for source in component
        for t in _effective_transitions(phases[source])
        if _phase_ref_name(t.target) in component
    ]
    if any(not isinstance(guard, Expr) for _, _, guard in internal):
        return CompileIssue(
            location=location,
            message="C2* inconclusive: cannot check non-symbolic internal guards",
        )
    guards = [(source, target, cast(Expr, guard)) for source, target, guard in internal]
    guard_vars = frozenset().union(*(g.variables for _, _, g in guards))
    writes = {
        name: frozenset(path for node in phases[name].nodes for path, _ in _node_ref_outputs(node))
        & guard_vars
        for name in component
    }
    relevant = frozenset().union(*writes.values())
    bound: int | None = len(component)
    for path in relevant:
        if path not in domains:
            bound = None
            break
        bound *= len(domains[path])
    limit = min(max_depth, depth if depth is not None else max_depth)
    if bound is not None:
        limit = min(limit, max(1, bound))
    solver = z3.Solver()
    state = {
        path: _z3_variable_for_type(output_types[path], f"state::0::{path}")
        for path in sorted(guard_vars)
    }
    ctx = Z3Context(output_types, domains=domains, bindings=state)
    solver.add(*ctx.domain_constraints(guard_vars))
    phase_ids = {name: i for i, name in enumerate(component)}
    current = z3.Int("phase::0")
    solver.add(z3.Or(*(current == i for i in phase_ids.values())))
    for step in range(1, limit + 1):
        following = z3.Int(f"phase::{step}")
        post = {
            path: _z3_variable_for_type(output_types[path], f"state::{step}::{path}")
            if path in relevant
            else state[path]
            for path in sorted(guard_vars)
        }
        ctx = Z3Context(output_types, domains=domains, bindings=post)
        solver.add(*ctx.domain_constraints(relevant))
        branches = []
        for source in component:
            branches.append(
                z3.And(
                    current == phase_ids[source],
                    *(post[path] == state[path] for path in sorted(relevant - writes[source])),
                    z3.Or(
                        *(
                            z3.And(following == phase_ids[cast(str, target)], guard.to_z3(ctx))
                            for origin, target, guard in guards
                            if origin == source
                        )
                    ),
                )
            )
        solver.add(z3.Or(*branches))
        result = solver.check()
        if result == z3.unsat:
            return None
        if result == z3.unknown:
            return CompileIssue(
                location=location,
                message=f"C2* inconclusive at depth {step}: solver UNKNOWN ({solver.reason_unknown()})",
            )
        state, current = post, following
    if bound is not None and limit >= bound:
        detail = (
            f"SAT at finite bound N_S={bound}; infinite local residence is possible, "
            "but global nontermination requires a reachable entry"
        )
    else:
        detail = f"SAT through depth {limit}; checking budget exhausted, termination unproved"
    return CompileIssue(location=location, message=f"C2*: {detail}")


def _check_c3_for_phase(
    phase: Phase,
    nodes: tuple[Node, ...],
    *,
    max_states: int = 4096,
) -> list[CompileIssue]:
    if not phase.transitions:
        return []
    effective_transitions = _effective_transitions(phase)
    if all(isinstance(transition.predicate, Expr) for transition in effective_transitions):
        return _check_c3_for_phase_with_z3(phase, nodes, effective_transitions)
    issues: list[CompileIssue] = []
    samples = _state_samples(nodes, max_states=max_states)
    if samples is None:
        return [
            CompileIssue(
                location=phase.name,
                message=f"C3 check skipped: finite state sample exceeds {max_states}",
            )
        ]

    for state in samples:
        enabled: list[str] = []
        for transition in effective_transitions:
            try:
                if _evaluate_guard(transition.predicate, state):
                    enabled.append(transition.name)
            except KeyError:
                return []
            except Exception as exc:
                issues.append(
                    CompileIssue(
                        location=f"{phase.name}.{transition.name}",
                        message=f"C3 predicate failed during compile check: {exc}",
                    )
                )
                continue
        if len(enabled) != 1:
            issues.append(
                CompileIssue(
                    location=phase.name,
                    message=(
                        "C3 violation: expected exactly one enabled transition, "
                        f"got {enabled or ['none']} at state {state}"
                    ),
                )
            )
            break
    return issues


def _check_c3_for_phase_with_z3(
    phase: Phase,
    nodes: tuple[Node, ...],
    transitions: tuple[EffectiveTransition, ...],
) -> list[CompileIssue]:
    output_types = _output_types(nodes)
    output_types[f"{SYSTEM_CLOCK}.tick"] = int
    output_types[f"{SYSTEM_CLOCK}.time"] = float
    domains = _finite_domains_by_path(nodes)
    ctx = Z3Context(output_types, domains=domains)
    try:
        predicates = [cast(Expr, transition.predicate).to_z3(ctx) for transition in transitions]
    except Exception as exc:
        return [
            CompileIssue(
                location=phase.name,
                message=f"C3 z3 check failed: {exc}",
            )
        ]

    issues: list[CompileIssue] = []
    guard_vars = frozenset().union(
        *[cast(Expr, transition.predicate).variables for transition in transitions]
    )
    for left_index, left in enumerate(predicates):
        for right_index, right in enumerate(predicates[left_index + 1 :], left_index + 1):
            solver = z3.Solver()
            solver.add(*ctx.domain_constraints(guard_vars))
            solver.add(left)
            solver.add(right)
            if solver.check() == z3.sat:
                left_transition = transitions[left_index]
                right_transition = transitions[right_index]
                issues.append(
                    CompileIssue(
                        location=phase.name,
                        message=(
                            "C3 violation: transitions "
                            f"{left_transition.name!r} and {right_transition.name!r} "
                            f"overlap at state {_model_snapshot(solver.model(), ctx)}"
                        ),
                    )
                )
                return issues

    solver = z3.Solver()
    solver.add(*ctx.domain_constraints(guard_vars))
    solver.add(z3.Not(z3.Or(*predicates)))
    if solver.check() == z3.sat:
        issues.append(
            CompileIssue(
                location=phase.name,
                message=(
                    "C3 violation: no transition is enabled at state "
                    f"{_model_snapshot(solver.model(), ctx)}"
                ),
            )
        )
    return issues


def _model_snapshot(
    model: z3.ModelRef,
    ctx: Z3Context,
    *,
    limit: int | None = None,
) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    variables = ctx.variables | ctx.bindings
    for index, (path, variable) in enumerate(sorted(variables.items())):
        if limit is not None and index >= limit:
            snapshot["..."] = f"{len(variables) - limit} more"
            break
        value = model.eval(variable, model_completion=True)
        snapshot[path] = str(value)
    return snapshot


def _output_types(nodes: tuple[Node, ...]) -> dict[str, type[Any]]:
    types: dict[str, type[Any]] = {}
    for node in nodes:
        node_cls = node.__class__
        module = sys.modules[node_cls.__module__]
        try:
            hints = get_type_hints(
                node_cls._output_namespace_cls,
                globalns=vars(module),
                localns={node_cls.__name__: node_cls},
                include_extras=True,
            )
        except Exception:
            hints = {}
        for name, output in node_cls._outputs.items():
            annotation = hints.get(name)
            types[_node_output_path(node, output)] = _base_type(annotation)
    return types


def _finite_domains_by_path(nodes: tuple[Node, ...]) -> dict[str, tuple[Any, ...]]:
    domains: dict[str, tuple[Any, ...]] = {}
    output_types = _output_types(nodes)
    for node in nodes:
        for output in node.__class__._outputs.values():
            path = _node_output_path(node, output)
            domain = output.finite_domain(node)
            if domain is None:
                output_type = output_types.get(path)
                if _is_enum_type(output_type):
                    domain = tuple(output_type)
            if domain is not None:
                domains[path] = domain
    return domains


def _base_type(annotation: Any) -> type[Any]:
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    if annotation in (bool, int, float):
        return annotation
    if _is_enum_type(annotation):
        return annotation
    return object


def _state_samples(
    nodes: tuple[Node, ...],
    *,
    max_states: int,
) -> list[StateSnapshot] | None:
    entries: list[tuple[str, tuple[Any, ...]]] = []
    state_count = 1
    domains_by_path = _finite_domains_by_path(nodes)
    for node in nodes:
        for output in node.__class__._outputs.values():
            path = _node_output_path(node, output)
            domain = domains_by_path.get(path)
            if domain is None:
                continue
            state_count *= len(domain)
            if state_count > max_states:
                return None
            entries.append((path, domain))

    keys = [key for key, _ in entries]
    domains = [domain for _, domain in entries]
    return [dict(zip(keys, values, strict=True)) for values in product(*domains)]
