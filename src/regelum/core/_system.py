from __future__ import annotations

from collections.abc import Iterable, Mapping
from fractions import Fraction
from typing import Any, cast

from regelum.core._base import (
    _MISSING,
    SYSTEM_CLOCK,
    SYSTEM_OUTPUTS,
    BaseTimeStep,
    BoundVarPort,
    CompileError,
    CompileIssue,
    CompileReport,
    Connection,
    Node,
    NodeInputs,
    NodeRef,
    NodeState,
    Phase,
    StateSnapshot,
    StepRecord,
    SystemSource,
    T,
    VarPort,
    VarSource,
    _accepts_keyword,
    _class_output_paths,
    _deduplicate_implicit_node_names,
    _evaluate_guard,
    _gcd_fraction,
    _node_input_path,
    _node_output_path,
    _normalize_output_path,
    _parse_time_step,
    _resolve_lazy_source,
)
from regelum.core._checks import (
    _check_c2star,
    _check_c3_for_phase,
    _check_clock_name_is_reserved,
    _check_continuous_phase_contract,
    _check_cross_ode_system_coupling,
    _check_phase_graph_completeness,
    _check_phase_guard_references,
    _check_phase_kinds,
    _check_phase_reachability,
    _check_transition_structure,
    _connection_map,
    _effective_transitions,
    _is_continuous_phase,
    _is_ode_system,
    _node_connections,
    _nodes_from_phases,
    _ode_system_dt,
    _ode_system_nodes,
    _phase_dependency_edges,
    _phase_ref_name,
    _required_initial_issues,
    _run_requires_inputs,
    _schedule_warnings,
    _topological_order,
    compile_nodes,
)


class PhasedReactiveSystem:
    def __init__(
        self,
        *,
        phases: Iterable[Phase],
        base_dt: BaseTimeStep = "auto",
        connections: Iterable[Connection] = (),
        initial_state: Mapping[Any, Any] | None = None,
        initial_phase: str | None = None,
        initial_tick: int = 0,
        max_phase_steps: int = 64,
        c2star_depth: int | None = None,
        c2star_max_depth: int = 64,
        strict: bool = True,
    ) -> None:
        self.phases = tuple(phases)
        self.nodes = _nodes_from_phases(self.phases)
        _deduplicate_implicit_node_names(self.nodes)
        self.connections = (*_node_connections(self.nodes), *tuple(connections))
        self._connection_map = _connection_map(self.connections)
        self.initial_phase = initial_phase or self._infer_initial_phase()
        self.max_phase_steps = max_phase_steps
        self.c2star_depth = c2star_depth
        self.c2star_max_depth = c2star_max_depth
        self.strict = strict
        self._requested_base_dt = base_dt
        self._continuous_phase_names = tuple(
            phase.name for phase in self.phases if _is_continuous_phase(phase)
        )
        self._base_dt = self._resolve_base_dt(base_dt)
        self._period_ticks = self._resolve_period_ticks()
        self._initial_tick = initial_tick
        self._tick = initial_tick
        self._time_tick = initial_tick
        self._nodes_by_id = {node.node_id: node for node in self.nodes}
        self._nodes_by_type = self._index_nodes_by_type()
        self._outputs_by_path = self._index_outputs()
        self._initial_state_overrides = self._resolve_initial_state(initial_state or {})
        self.compile_report = self._compile()
        if strict and not self.compile_report.ok:
            raise CompileError(self.compile_report)
        self._phases_by_name = {phase.name: phase for phase in self.phases}
        self._phase_schedules = dict(self.compile_report.phase_schedules)
        self._state: dict[str, Any] = {}
        self._history: list[StepRecord] = []
        self.reset()

    @property
    def history(self) -> tuple[StepRecord, ...]:
        return tuple(self._history)

    @property
    def base_dt(self) -> Fraction:
        return self._base_dt

    @property
    def tick(self) -> int:
        return self._tick

    def _resolve_base_dt(self, requested: BaseTimeStep) -> Fraction:
        if requested != "auto":
            return _parse_time_step(requested, field_name="base_dt")
        explicit_dts = self._explicit_schedule_dts()
        if not any(_is_ode_system(node) for node in self.nodes):
            return Fraction(1, 1)
        return _gcd_fraction(explicit_dts)

    def _explicit_schedule_dts(self) -> tuple[Fraction, ...]:
        values: list[Fraction] = []
        for node in self.nodes:
            if _is_ode_system(node):
                values.append(_ode_system_dt(node))
                continue
            node_dt = getattr(node, "_schedule_dt", None)
            if node_dt is not None:
                values.append(node_dt)
        return tuple(values)

    def _resolve_period_ticks(self) -> dict[str, int]:
        periods: dict[str, int] = {}
        for node in self.nodes:
            effective_dt = getattr(node, "_schedule_dt", None) or self._base_dt
            if _is_ode_system(node):
                effective_dt = _ode_system_dt(node)
            ratio = effective_dt / self._base_dt
            if ratio.denominator != 1:
                raise ValueError(
                    f"{node.node_id}.dt={effective_dt} is not an integer multiple "
                    f"of base_dt={self._base_dt}."
                )
            periods[node.node_id] = ratio.numerator
        return periods

    def _is_due(self, node: Node) -> bool:
        return self._tick % self._period_ticks[node.node_id] == 0

    def _clock_state(self) -> dict[str, Any]:
        return {
            f"{SYSTEM_CLOCK}.tick": self._tick,
            f"{SYSTEM_CLOCK}.time": float(self._time_tick * self._base_dt),
        }

    def _commit_clock(self) -> None:
        self._state.update(self._clock_state())

    def reset(
        self,
        initial_state: Mapping[Any, Any] | None = None,
    ) -> None:
        self._state.clear()
        self._history.clear()
        self._tick = self._initial_tick
        self._time_tick = self._initial_tick
        self._reset_ode_runtime_state()
        self._commit_clock()
        for node in self.nodes:
            for output in node.__class__._outputs.values():
                if output.initial is _MISSING:
                    continue
                try:
                    self._state[_node_output_path(node, output)] = output.initial_value(node)
                except Exception:
                    if self.strict:
                        raise
        if initial_state is None:
            self._state.update(self._initial_state_overrides)
        else:
            self._state.update(self._resolve_initial_state(initial_state))
        self._sync_ode_runtime_state_from_system_state()
        self._commit_clock()

    def read(self, output: VarSource[T]) -> T:
        path = self._resolve_output(output)
        return self._state[path]

    def snapshot(self) -> dict[str, Any]:
        return {path: value for path, value in self._state.items() if path not in SYSTEM_OUTPUTS}

    def _resolve_initial_state(
        self,
        initial_state: Mapping[Any, Any],
    ) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        for output, value in initial_state.items():
            resolved[self._resolve_output(output)] = value
        return resolved

    def step(self) -> tuple[StepRecord, ...]:
        records: list[StepRecord] = []
        phase_name: str | None = self.initial_phase
        phase_steps = 0
        crossed_continuous_phase = False
        while phase_name is not None:
            phase_steps += 1
            if phase_steps > self.max_phase_steps:
                raise RuntimeError(f"Tick exceeded max_phase_steps={self.max_phase_steps}.")
            phase = self._phases_by_name[phase_name]
            if _is_continuous_phase(phase):
                records.extend(self._run_continuous_phase(phase))
                crossed_continuous_phase = True
            else:
                for node_id in self._phase_schedules[phase.name]:
                    node = self._nodes_by_id[node_id]
                    if not self._is_due(node):
                        continue
                    record = self._run_discrete_node(phase, node)
                    records.append(record)
                    self._history.append(record)
            phase_name = self._choose_next_phase(phase)
        if crossed_continuous_phase:
            self._tick += 1
        else:
            self._tick += 1
            self._time_tick += 1
        self._commit_clock()
        return tuple(records)

    def _run_discrete_node(self, phase: Phase, node: Node) -> StepRecord:
        inputs = self._build_inputs(node)
        result = _run_node(node, inputs, state_snapshot=dict(self._state))
        state_vars = self._normalize_outputs(node, result)
        self._commit_node_outputs(node, state_vars)
        return StepRecord(
            phase=phase.name,
            node=node.__class__.__name__,
            inputs=dict(vars(inputs)),
            state=state_vars,
        )

    def _run_continuous_phase(self, phase: Phase) -> tuple[StepRecord, ...]:
        records: list[StepRecord] = []
        time_start = float(self._time_tick * self._base_dt)
        time_stop = time_start + float(self._base_dt)
        for node in phase.nodes:
            inputs = self._build_inputs(node)
            result = _run_node(
                node,
                inputs,
                state_snapshot=dict(self._state),
                time_start=time_start,
                time_stop=time_stop,
            )
            state_vars = self._normalize_outputs(node, result)
            self._commit_node_outputs(node, state_vars)
            self._commit_ode_state_outputs(node)
            record = StepRecord(
                phase=phase.name,
                node=node.__class__.__name__,
                inputs=dict(vars(inputs)),
                state=state_vars,
            )
            records.append(record)
            self._history.append(record)
        self._time_tick += 1
        self._commit_clock()
        return tuple(records)

    def _commit_node_outputs(self, node: Node, outputs: dict[str, Any]) -> None:
        for name, value in outputs.items():
            output = node.__class__._outputs[name]
            self._state[_node_output_path(node, output)] = value

    def _commit_ode_state_outputs(self, node: Node) -> None:
        if not _is_ode_system(node):
            return
        for ode_node in _ode_system_nodes(node):
            for name, value in ode_node._ode_state_values.items():
                port = ode_node.__class__._outputs[name]
                self._state[_node_output_path(ode_node, port)] = value

    def _reset_ode_runtime_state(self) -> None:
        time_s = float(self._time_tick * self._base_dt)
        for node in self.nodes:
            if _is_ode_system(node):
                cast(Any, node)._time_s = time_s
                continue
            state_vars = getattr(node.__class__, "_state_vars", {})
            if state_vars:
                cast(Any, node)._ode_state_values = {
                    name: port.initial_value(node) for name, port in state_vars.items()
                }

    def _sync_ode_runtime_state_from_system_state(self) -> None:
        for node in self.nodes:
            state_vars = getattr(node.__class__, "_state_vars", {})
            if not state_vars:
                continue
            values = cast(Any, node)._ode_state_values
            for name, port in state_vars.items():
                path = _node_output_path(node, port)
                if path in self._state:
                    values[name] = self._state[path]

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

    def compile(self) -> CompileReport:
        return self._compile()

    def _compile(self) -> CompileReport:
        report = compile_nodes(self.nodes, self.connections)
        issues = list(report.issues)
        warnings = list(report.warnings)
        issues.extend(_check_clock_name_is_reserved(self.nodes))
        phase_kind_issues = _check_phase_kinds(self.phases)
        issues.extend(phase_kind_issues)
        if len(self._continuous_phase_names) > 1:
            issues.append(
                CompileIssue(
                    location="phases",
                    message=(
                        "PhasedReactiveSystem supports at most one phase containing ODESystem "
                        "nodes for now"
                    ),
                )
            )
        issues.extend(
            _check_continuous_phase_contract(
                self.phases,
                self.nodes,
                initial_phase=self.initial_phase,
                max_phase_steps=self.max_phase_steps,
            )
        )
        issues.extend(_check_cross_ode_system_coupling(self.phases, report.inputs))
        warnings.extend(
            _schedule_warnings(
                base_dt=self._base_dt,
                requested_base_dt=self._requested_base_dt,
                continuous=bool(self._continuous_phase_names),
                explicit_dts=tuple(
                    getattr(node, "_schedule_dt")
                    for node in self.nodes
                    if getattr(node, "_schedule_dt", None) is not None and not _is_ode_system(node)
                ),
            )
        )
        completeness_issues = _check_phase_graph_completeness(
            self.phases,
            self.nodes,
            self._connection_map,
        )
        incomplete_source_locations = {
            issue.location for issue in completeness_issues if " reads " in issue.message
        }
        issues = [
            issue
            for issue in issues
            if not (
                issue.location in incomplete_source_locations
                and issue.message.startswith("unknown input source")
            )
        ]
        issues.extend(completeness_issues)
        phase_names = [phase.name for phase in self.phases]
        phase_name_set = set(phase_names)
        initial_phases = [phase.name for phase in self.phases if phase.is_initial]
        phase_schedules: dict[str, tuple[str, ...]] = {}
        phase_dependency_edges: dict[str, tuple[tuple[str, str], ...]] = {}

        for phase_name in sorted(name for name in phase_name_set if phase_names.count(name) > 1):
            issues.append(
                CompileIssue(
                    location=phase_name,
                    message="phase is declared more than once",
                )
            )
        if self.initial_phase not in phase_name_set:
            issues.append(
                CompileIssue(
                    location=self.initial_phase,
                    message="initial phase is not declared",
                )
            )
        if len(initial_phases) != 1:
            issues.append(
                CompileIssue(
                    location="phases",
                    message="exactly one phase must be marked initial",
                )
            )
        elif self.initial_phase != initial_phases[0]:
            issues.append(
                CompileIssue(
                    location=self.initial_phase,
                    message=(
                        f"initial_phase must match the phase marked initial ({initial_phases[0]!r})"
                    ),
                )
            )
        issues.extend(_check_phase_reachability(self.phases, self.initial_phase))
        issues.extend(
            _check_c2star(
                self.phases,
                self.nodes,
                depth=self.c2star_depth,
                max_depth=self.c2star_max_depth,
            )
        )
        for phase in self.phases:
            phase_node_ids = self._phase_node_ids(phase)
            if not phase.transitions:
                issues.append(
                    CompileIssue(
                        location=phase.name,
                        message="phase has no transitions",
                    )
                )
            for transition in phase.transitions:
                target = _phase_ref_name(transition.target)
                if target is not None and target not in phase_name_set:
                    issues.append(
                        CompileIssue(
                            location=f"{phase.name}.{transition.name}",
                            message=f"unknown phase target {target!r}",
                        )
                    )
            transition_issues, transition_warnings = _check_transition_structure(phase)
            issues.extend(transition_issues)
            warnings.extend(transition_warnings)
            guard_issues = _check_phase_guard_references(
                phase,
                self._outputs_by_path,
                _class_output_paths(self.nodes),
            )
            issues.extend(
                issue
                for issue in guard_issues
                if not (
                    issue.location in incomplete_source_locations
                    and issue.message.startswith("unknown guard variable")
                )
            )
            if not transition_warnings:
                issues.extend(_check_c3_for_phase(phase, self.nodes))
            dependency_edges = _phase_dependency_edges(phase_node_ids, report.inputs)
            phase_dependency_edges[phase.name] = tuple(dependency_edges)
            schedule = _topological_order(phase_node_ids, dependency_edges)
            if schedule is None:
                edge_text = ", ".join(f"{source}->{target}" for source, target in dependency_edges)
                issues.append(
                    CompileIssue(
                        location=phase.name,
                        message=(f"C1 violation: phase dependency graph is cyclic ({edge_text})"),
                    )
                )
                phase_schedules[phase.name] = phase_node_ids
            else:
                phase_schedules[phase.name] = schedule
        required_initial_state_vars = self._required_initial_state_vars(
            report.inputs,
            phase_schedules,
        )
        state_vars_without_initial = set(report.state_vars_without_initial)
        initial_state_vars = set(self._initial_state_overrides)
        issues.extend(
            _required_initial_issues(
                {
                    path: readers
                    for path, readers in required_initial_state_vars.items()
                    if path in state_vars_without_initial and path not in initial_state_vars
                }
            )
        )
        return CompileReport(
            nodes=report.nodes,
            inputs=report.inputs,
            state_vars=report.state_vars,
            issues=tuple(issues),
            warnings=tuple(warnings),
            phase_schedules=phase_schedules,
            phase_dependency_edges=phase_dependency_edges,
            state_vars_without_initial=report.state_vars_without_initial,
            required_initial_state_vars=required_initial_state_vars,
        )

    def _required_initial_state_vars(
        self,
        inputs: dict[str, str],
        phase_schedules: dict[str, tuple[str, ...]],
    ) -> dict[str, tuple[str, ...]]:
        issues: dict[str, set[str]] = {}
        phase_by_name = {phase.name: phase for phase in self.phases}
        node_state_vars = {
            node.node_id: tuple(
                _node_output_path(node, output) for output in node.__class__._outputs.values()
            )
            for node in self.nodes
        }
        node_inputs = {
            node.node_id: tuple(
                _node_input_path(node, input_port) for input_port in node.__class__._inputs.values()
            )
            for node in self.nodes
        }
        seen: set[tuple[str, frozenset[str]]] = set()
        stack: list[tuple[str, frozenset[str]]] = [(self.initial_phase, frozenset())]

        while stack:
            phase_name, written_before_phase = stack.pop()
            state_key = (phase_name, written_before_phase)
            if state_key in seen:
                continue
            seen.add(state_key)
            phase = phase_by_name.get(phase_name)
            if phase is None:
                continue

            written = set(written_before_phase)
            for node_id in phase_schedules.get(phase.name, self._phase_node_ids(phase)):
                for input_path in node_inputs.get(node_id, ()):
                    source_path = inputs.get(input_path)
                    if (
                        source_path is not None
                        and source_path not in SYSTEM_OUTPUTS
                        and source_path not in written
                    ):
                        issues.setdefault(source_path, set()).add(input_path)
                written.update(node_state_vars.get(node_id, ()))

            for transition in phase.transitions:
                target = _phase_ref_name(transition.target)
                if target is not None:
                    stack.append((target, frozenset(written)))

        return {
            source_path: tuple(sorted(readers)) for source_path, readers in sorted(issues.items())
        }

    def _infer_initial_phase(self) -> str:
        initial_phases = [phase.name for phase in self.phases if phase.is_initial]
        if len(initial_phases) == 1:
            return initial_phases[0]
        return "__invalid_initial_phase__"

    def _index_outputs(self) -> dict[str, str]:
        outputs: dict[str, str] = {}
        class_paths: dict[str, list[str]] = {}
        for node in self.nodes:
            for output in node.__class__._outputs.values():
                path = _node_output_path(node, output)
                outputs[path] = path
                class_paths.setdefault(output.path, []).append(path)
        for class_path, paths in class_paths.items():
            if len(paths) == 1:
                outputs[class_path] = paths[0]
        return outputs

    def _index_nodes_by_type(self) -> dict[type[Node], Node]:
        nodes_by_type: dict[type[Node], Node] = {}
        for node in self.nodes:
            if sum(1 for candidate in self.nodes if candidate.__class__ is node.__class__) == 1:
                nodes_by_type[node.__class__] = node
        return nodes_by_type

    def _resolve_node_ref(self, node_ref: NodeRef) -> Node:
        if isinstance(node_ref, Node):
            return node_ref
        try:
            return self._nodes_by_type[node_ref]
        except KeyError as exc:
            raise KeyError(
                f"Phase node reference {node_ref.__name__} is ambiguous or unknown; "
                "use a node instance instead."
            ) from exc

    def _phase_node_ids(self, phase: Phase) -> tuple[str, ...]:
        return tuple(node.node_id for node in phase.nodes)

    def _resolve_output(self, output: VarSource[T]) -> str:
        output = _resolve_lazy_source(output)
        if isinstance(output, BoundVarPort):
            return output.path
        if isinstance(output, VarPort):
            output = output.path
        if isinstance(output, SystemSource):
            output = output.path
        output = _normalize_output_path(output)
        if "." not in output:
            raise ValueError(f"State reference must be 'Node.var', got {output!r}.")
        if output in SYSTEM_OUTPUTS:
            return output
        try:
            return self._outputs_by_path[output]
        except KeyError as exc:
            raise KeyError(f"Unknown state reference: {output}") from exc

    def _choose_next_phase(self, phase: Phase) -> str | None:
        snapshot = dict(self._state)
        enabled = [
            transition
            for transition in _effective_transitions(phase)
            if _evaluate_guard(transition.predicate, snapshot)
        ]
        if len(enabled) != 1:
            names = [transition.name for transition in enabled]
            raise RuntimeError(
                f"Phase {phase.name!r} must enable exactly one transition; "
                f"enabled={names or ['none']}."
            )
        return _phase_ref_name(enabled[0].target)

    def _build_inputs(self, node: Node) -> NodeInputs:
        values: dict[str, Any] = {}
        for name, input_port in node.__class__._inputs.items():
            input_path = _node_input_path(node, input_port)
            source_ref = self._connection_map.get(input_path, input_port.source)
            if source_ref is None:
                raise RuntimeError(f"src {input_path} is not connected.")
            output = self._resolve_output(cast(VarSource[Any], source_ref))
            value = self._state.get(output, input_port.default)
            values[name] = value
        return node.__class__.Inputs(**values)

    def _normalize_outputs(
        self,
        node: Node,
        result: NodeState | dict[str, Any],
    ) -> dict[str, Any]:
        state_vars = dict(result) if isinstance(result, dict) else dict(vars(result))
        declared = node.__class__._outputs
        unknown = set(state_vars) - set(declared)
        if unknown:
            raise ValueError(
                f"{node.__class__.__name__}.update returned undeclared state variables: {sorted(unknown)}"
            )
        missing = set(declared) - set(state_vars)
        if missing:
            raise ValueError(
                f"{node.__class__.__name__}.update did not return state variables: {sorted(missing)}"
            )
        return state_vars


def _run_node(
    node: Node,
    inputs: NodeInputs,
    *,
    state_snapshot: StateSnapshot | None = None,
    time_start: float | None = None,
    time_stop: float | None = None,
) -> Any:
    kwargs: dict[str, Any] = {}
    if state_snapshot is not None and _accepts_keyword(node.update, "state_snapshot"):
        kwargs["state_snapshot"] = state_snapshot
    for name in node.__class__._update_state_parameter_names:
        kwargs[name] = _previous_node_state(node, state_snapshot)
    if time_start is not None and _accepts_keyword(node.update, "time_start"):
        kwargs["time_start"] = time_start
    if time_stop is not None and _accepts_keyword(node.update, "time_stop"):
        kwargs["time_stop"] = time_stop
    if node.__class__._run_input_mode == "parameters":
        return node.update(**vars(inputs), **kwargs)
    if node.__class__._inputs or _run_requires_inputs(node):
        return node.update(inputs, **kwargs)
    return node.update(**kwargs)


def _previous_node_state(node: Node, state_snapshot: StateSnapshot | None) -> NodeState:
    if state_snapshot is None:
        raise RuntimeError(
            f"{node.__class__.__name__}.update requested previous State, "
            "but no state snapshot is available."
        )
    values: dict[str, Any] = {}
    for name, port in node.__class__._outputs.items():
        path = _node_output_path(node, port)
        try:
            values[name] = state_snapshot[path]
        except KeyError as exc:
            raise RuntimeError(
                f"{node.__class__.__name__}.update requested previous State, "
                f"but {path!r} is not available; define var(init=...) or pass initial_state."
            ) from exc
    state_cls = cast(type[NodeState], node.__class__._output_namespace_cls)
    return state_cls(**values)
