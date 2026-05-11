from typing import cast

import pytest

from regelum import (
    CompileError,
    Elif,
    Else,
    ElseIf,
    Goto,
    If,
    Input,
    Node,
    NodeInputs,
    NodeState,
    Phase,
    PhasedReactiveSystem,
    V,
    Var,
    port,
    terminate,
)


class Source(Node):
    class State(NodeState):
        value: int = Var(init=1)
        ready: bool = Var(init=True)

    def __init__(
        self,
        value: int = 1,
        ready: bool = True,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.value = value
        self.ready = ready

    def update(self) -> State:
        return self.State(value=self.value, ready=self.ready)


class Sink(Node):
    class Inputs(NodeInputs):
        value: int = Input(src=Source.State.value)

    class State(NodeState):
        seen: int = Var(init=0)

    def update(self, inputs: Inputs) -> State:
        return self.State(seen=inputs.value)


class UnconnectedSink(Node):
    class Inputs(NodeInputs):
        value: int = Input()

    class State(NodeState):
        seen: int = Var(init=0)

    def update(self, inputs: Inputs) -> State:
        return self.State(seen=inputs.value)


class Worker(Node):
    class State(NodeState):
        ran: bool = Var(init=False)

    def update(self) -> State:
        return self.State(ran=True)


class Flag(Node):
    class State(NodeState):
        ready: bool = Var(init=True)
        blocked: bool = Var(init=False)
        level: int = Var(init=0)

    def __init__(
        self,
        *,
        ready: bool = True,
        blocked: bool = False,
        level: int = 0,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.ready = ready
        self.blocked = blocked
        self.level = level

    def update(self) -> State:
        return self.State(
            ready=self.ready,
            blocked=self.blocked,
            level=self.level,
        )


class LinkA(Node):
    class State(NodeState):
        value: int = Var(init=1)


class LinkB(Node):
    class Inputs(NodeInputs):
        value: int = Input(src=LinkA.State.value)

    class State(NodeState):
        value: int = Var(init=0)

    def update(self, inputs: Inputs) -> State:
        return self.State(value=inputs.value)


class LinkC(Node):
    class Inputs(NodeInputs):
        value: int = Input(src=LinkB.State.value)

    class State(NodeState):
        value: int = Var(init=0)

    def update(self, inputs: Inputs) -> State:
        return self.State(value=inputs.value)


def _single_phase_system(*nodes: Node, strict: bool = True) -> PhasedReactiveSystem:
    return PhasedReactiveSystem(
        phases=[
            Phase(
                "tick",
                nodes=nodes,
                transitions=(Goto(terminate),),
                is_initial=True,
            )
        ],
        strict=strict,
    )


def _messages(error: CompileError) -> tuple[str, ...]:
    return tuple(issue.message for issue in error.report.issues)


def test_var_uses_init_keyword_without_initial_alias() -> None:
    old_keyword = "initial"
    with pytest.raises(TypeError, match="unexpected keyword argument 'initial'"):
        Var(**{old_keyword: 0})


def test_update_can_receive_previous_state_without_declaring_self_input() -> None:
    class Counter(Node):
        class State(NodeState):
            count: int = Var(init=0)

        def update(self, prev_state: State) -> State:
            return self.State(count=prev_state.count + 1)

    counter = Counter()
    system = _single_phase_system(counter)

    assert system.compile_report.ok
    assert system.compile_report.inputs == {}
    system.run(steps=3)
    assert system.read(counter.State.count) == 3


def test_update_can_receive_inputs_object_and_previous_state() -> None:
    class Accumulator(Node):
        class Inputs(NodeInputs):
            value: int = Input(src=Source.State.value)

        class State(NodeState):
            total: int = Var(init=0)

        def update(self, inputs: Inputs, state: State) -> State:
            return self.State(total=state.total + inputs.value)

    source = Source(value=2)
    accumulator = Accumulator()
    system = _single_phase_system(source, accumulator)

    assert system.compile_report.ok
    assert system.compile_report.inputs["Accumulator.value"] == "Source.value"
    system.run(steps=2)
    assert system.read(accumulator.State.total) == 4


def test_update_can_receive_previous_state_from_deferred_annotation() -> None:
    class Counter(Node):
        class State(NodeState):
            count: int = Var(init=0)

        def update(self, prev_state: "State") -> "State":
            return self.State(count=prev_state.count + 1)

    counter = Counter()
    system = _single_phase_system(counter)

    system.run(steps=2)
    assert system.read(counter.State.count) == 2


def test_update_can_receive_parameter_inputs_and_previous_state() -> None:
    class Accumulator(Node):
        class State(NodeState):
            total: int = Var(init=0)

        def update(
            self,
            value: int = Input(src=Source.State.value),
            *,
            prevstate: State,
        ) -> State:
            return self.State(total=prevstate.total + value)

    source = Source(value=3)
    accumulator = Accumulator()
    system = _single_phase_system(source, accumulator)

    assert system.compile_report.ok
    assert system.compile_report.inputs["Accumulator.value"] == "Source.value"
    system.run(steps=2)
    assert system.read(accumulator.State.total) == 6


def test_previous_state_requires_initial_value_or_initial_state() -> None:
    class Counter(Node):
        class State(NodeState):
            count: int = Var()

        def update(self, prev_state: State) -> State:
            return self.State(count=prev_state.count + 1)

    counter = Counter()
    system = _single_phase_system(counter)

    with pytest.raises(RuntimeError, match="define Var\\(init=\\.\\.\\.\\) or pass initial_state"):
        system.step()

    system.reset(initial_state={counter.State.count: 10})
    system.step()
    assert system.read(counter.State.count) == 11


def test_class_level_input_ref_resolves_when_single_instance_exists() -> None:
    source = Source(value=7)
    sink = Sink()

    system = _single_phase_system(sink, source)

    assert system.compile_report.ok
    assert system.compile_report.inputs["Sink.value"] == "Source.value"
    assert system.compile_report.phase_schedules["tick"] == ("Source", "Sink")
    system.step()
    assert system.snapshot()["Sink.seen"] == 7


def test_class_level_input_ref_resolves_across_phases_when_single_instance_exists() -> None:
    source = Source(value=7)
    sink = Sink()

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "produce",
                nodes=(source,),
                transitions=(Goto("consume"),),
                is_initial=True,
            ),
            Phase("consume", nodes=(sink,), transitions=(Goto(terminate),)),
        ],
    )

    assert system.compile_report.ok
    assert system.compile_report.inputs["Sink.value"] == "Source.value"
    system.run(steps=1)
    assert system.snapshot()["Sink.seen"] == 7


def test_class_level_input_ref_is_ambiguous_with_two_instances_in_one_phase() -> None:
    with pytest.raises(CompileError) as exc_info:
        _single_phase_system(Source(), Source(), Sink())

    assert any(
        "ambiguous input source 'Source.value'" in message and "use instance connection" in message
        for message in _messages(exc_info.value)
    )


def test_class_level_input_ref_is_ambiguous_with_two_instances_across_phases() -> None:
    sink = Sink()

    with pytest.raises(CompileError) as exc_info:
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "produce-a",
                    nodes=(Source(name="source_a"),),
                    transitions=(Goto("produce-b"),),
                    is_initial=True,
                ),
                Phase(
                    "produce-b",
                    nodes=(Source(name="source_b"),),
                    transitions=(Goto("consume"),),
                ),
                Phase("consume", nodes=(sink,), transitions=(Goto(terminate),)),
            ],
        )

    assert any(
        "ambiguous input source 'Source.value'" in message
        and "source_a.value" in message
        and "source_b.value" in message
        for message in _messages(exc_info.value)
    )


def test_instance_bound_input_ref_selects_one_of_two_instances() -> None:
    source_a = Source(value=1)
    source_b = Source(value=2)
    sink = UnconnectedSink()
    port(sink.Inputs.value).connect(source_b.State.value)

    system = _single_phase_system(source_a, source_b, sink)

    assert system.compile_report.ok
    assert system.compile_report.inputs["UnconnectedSink.value"] == "Source_2.value"
    system.step()
    assert system.snapshot()["UnconnectedSink.seen"] == 2


def test_instance_bound_input_ref_may_point_to_node_in_another_phase() -> None:
    source = Source(value=3)
    sink = UnconnectedSink()
    port(sink.Inputs.value).connect(source.State.value)

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "produce",
                nodes=(source,),
                transitions=(Goto("consume"),),
                is_initial=True,
            ),
            Phase("consume", nodes=(sink,), transitions=(Goto(terminate),)),
        ],
    )

    assert system.compile_report.ok
    assert system.compile_report.inputs["UnconnectedSink.value"] == "Source.value"


def test_instance_bound_input_ref_to_node_outside_phases_is_incomplete_graph() -> None:
    source = Source(value=3)
    sink = UnconnectedSink()
    port(sink.Inputs.value).connect(source.State.value)

    system = _single_phase_system(sink, strict=False)

    assert any(
        issue.location == "UnconnectedSink.value"
        and "incomplete phase graph" in issue.message
        and "Source.value" in issue.message
        and "Source is not assigned to any phase" in issue.message
        for issue in system.compile_report.issues
    )
    assert not any(
        issue.location == "UnconnectedSink.value" and "unknown input source" in issue.message
        for issue in system.compile_report.issues
    )


def test_input_closure_is_not_auto_completed_from_missing_producers() -> None:
    system = _single_phase_system(LinkC(), strict=False)

    assert any(
        issue.location == "LinkC.value" and "unknown input source 'LinkB.value'" in issue.message
        for issue in system.compile_report.issues
    )
    assert tuple(system.compile_report.nodes) == ("LinkC",)


def test_input_closure_checks_second_hop_when_intermediate_node_is_covered() -> None:
    system = _single_phase_system(LinkB(), LinkC(), strict=False)

    assert any(
        issue.location == "LinkB.value" and "unknown input source 'LinkA.value'" in issue.message
        for issue in system.compile_report.issues
    )
    assert not any(issue.location == "LinkC.value" for issue in system.compile_report.issues)


def test_phase_nodes_accept_instances_only() -> None:
    with pytest.raises(TypeError, match="Phase.nodes accepts node instances only"):
        Phase(
            "bad",
            nodes=(cast(Node, Source),),
            transitions=(Goto(terminate),),
            is_initial=True,
        )


def test_duplicate_implicit_names_do_not_break_instance_input_refs() -> None:
    source_a = Source(value=1)
    source_b = Source(value=2)
    sink = UnconnectedSink()
    port(sink.Inputs.value).connect(source_b.State.value)

    system = _single_phase_system(source_a, source_b, sink)

    assert system.compile_report.nodes == ("Source", "Source_2", "UnconnectedSink")
    assert system.compile_report.inputs["UnconnectedSink.value"] == "Source_2.value"


def test_duplicate_explicit_names_are_rejected() -> None:
    with pytest.raises(CompileError) as exc_info:
        _single_phase_system(Source(name="source"), Source(name="source"))

    assert any(
        issue.location == "source" and issue.message == "node name is declared more than once"
        for issue in exc_info.value.report.issues
    )


def test_guard_only_node_is_present_in_inferred_node_graph() -> None:
    flag = Flag(name="flag")
    worker = Worker()

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "work",
                nodes=(worker, flag),
                transitions=(
                    If(V(flag.State.ready), terminate, name="ready"),
                    Else(terminate, name="not-ready"),
                ),
                is_initial=True,
            )
        ],
    )

    assert system.compile_report.ok
    assert system.compile_report.nodes == ("Worker", "flag")
    assert tuple(node.node_id for node in system.nodes) == ("Worker", "flag")


def test_instance_guard_source_may_point_to_node_in_another_phase() -> None:
    flag = Flag(name="flag")
    worker = Worker()

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "observe",
                nodes=(flag,),
                transitions=(Goto("work"),),
                is_initial=True,
            ),
            Phase(
                "work",
                nodes=(worker,),
                transitions=(
                    If(V(flag.State.ready), terminate, name="ready"),
                    Else(terminate, name="not-ready"),
                ),
            ),
        ],
    )

    assert system.compile_report.ok


def test_instance_guard_source_outside_all_phases_is_incomplete_graph() -> None:
    flag = Flag(name="external_flag")
    worker = Worker()

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "work",
                nodes=(worker,),
                transitions=(
                    If(V(flag.State.ready), terminate, name="ready"),
                    Else(terminate, name="not-ready"),
                ),
                is_initial=True,
            )
        ],
        strict=False,
    )

    assert any(
        issue.location == "work.ready"
        and "incomplete phase graph" in issue.message
        and "external_flag.ready" in issue.message
        and "external_flag is not assigned to any phase" in issue.message
        for issue in system.compile_report.issues
    )
    assert not any(
        issue.location == "work.ready" and "unknown guard variable" in issue.message
        for issue in system.compile_report.issues
    )


def test_class_level_guard_ref_resolves_when_single_instance_exists_in_same_phase() -> None:
    flag = Flag()

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "check",
                nodes=(flag,),
                transitions=(
                    If(V(Flag.State.ready), terminate, name="ready"),
                    Else(terminate, name="not-ready"),
                ),
                is_initial=True,
            )
        ],
    )

    assert system.compile_report.ok


def test_class_level_guard_ref_resolves_when_single_instance_exists_in_another_phase() -> None:
    flag = Flag()
    worker = Worker()

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "observe",
                nodes=(flag,),
                transitions=(Goto("work"),),
                is_initial=True,
            ),
            Phase(
                "work",
                nodes=(worker,),
                transitions=(
                    If(V(Flag.State.ready), terminate, name="ready"),
                    Else(terminate, name="not-ready"),
                ),
            ),
        ],
    )

    assert system.compile_report.ok


def test_class_level_guard_ref_is_ambiguous_with_two_instances_in_one_phase() -> None:
    with pytest.raises(CompileError) as exc_info:
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "check",
                    nodes=(Flag(name="flag_a"), Flag(name="flag_b")),
                    transitions=(
                        If(V(Flag.State.ready), terminate, name="ready"),
                        Else(terminate, name="not-ready"),
                    ),
                    is_initial=True,
                )
            ],
        )

    assert any(
        issue.location == "check.ready"
        and "ambiguous guard variable 'Flag.ready'" in issue.message
        and "flag_a.ready" in issue.message
        and "flag_b.ready" in issue.message
        and "use instance state reference" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_class_level_guard_ref_is_ambiguous_with_two_instances_across_phases() -> None:
    with pytest.raises(CompileError) as exc_info:
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "observe-a",
                    nodes=(Flag(name="flag_a"),),
                    transitions=(Goto("observe-b"),),
                    is_initial=True,
                ),
                Phase(
                    "observe-b",
                    nodes=(Flag(name="flag_b"),),
                    transitions=(Goto("check"),),
                ),
                Phase(
                    "check",
                    nodes=(Worker(),),
                    transitions=(
                        If(V(Flag.State.ready), terminate, name="ready"),
                        Else(terminate, name="not-ready"),
                    ),
                ),
            ],
        )

    assert any(
        issue.location == "check.ready"
        and "ambiguous guard variable 'Flag.ready'" in issue.message
        and "flag_a.ready" in issue.message
        and "flag_b.ready" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_instance_bound_guard_ref_selects_one_of_two_instances() -> None:
    flag_a = Flag(name="flag_a")
    flag_b = Flag(name="flag_b")

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "check",
                nodes=(flag_a, flag_b),
                transitions=(
                    If(V(flag_b.State.ready), terminate, name="ready"),
                    Else(terminate, name="not-ready"),
                ),
                is_initial=True,
            )
        ],
    )

    assert system.compile_report.ok


def test_instance_bound_guard_ref_selects_instance_across_phases() -> None:
    flag_a = Flag(name="flag_a")
    flag_b = Flag(name="flag_b")

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "observe-a",
                nodes=(flag_a,),
                transitions=(Goto("observe-b"),),
                is_initial=True,
            ),
            Phase(
                "observe-b",
                nodes=(flag_b,),
                transitions=(Goto("check"),),
            ),
            Phase(
                "check",
                nodes=(Worker(),),
                transitions=(
                    If(V(flag_b.State.ready), terminate, name="ready"),
                    Else(terminate, name="not-ready"),
                ),
            ),
        ],
    )

    assert system.compile_report.ok


def test_complex_guard_expression_collects_all_instance_sources() -> None:
    flag_a = Flag(name="flag_a", ready=True)
    flag_b = Flag(name="flag_b", blocked=False)
    flag_c = Flag(name="flag_c", level=5)

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "check",
                nodes=(flag_a, flag_b, flag_c),
                transitions=(
                    If(
                        (V(flag_a.State.ready) & ~V(flag_b.State.blocked))
                        | (V(flag_c.State.level) > 3),
                        terminate,
                        name="complex",
                    ),
                    Else(terminate, name="fallback"),
                ),
                is_initial=True,
            )
        ],
    )

    assert system.compile_report.ok


def test_complex_guard_expression_reports_missing_instance_source() -> None:
    flag_a = Flag(name="flag_a", ready=True)
    flag_b = Flag(name="flag_b", blocked=False)
    flag_c = Flag(name="flag_c", level=5)

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "check",
                nodes=(flag_a, flag_c),
                transitions=(
                    If(
                        (V(flag_a.State.ready) & ~V(flag_b.State.blocked))
                        | (V(flag_c.State.level) > 3),
                        terminate,
                        name="complex",
                    ),
                    Else(terminate, name="fallback"),
                ),
                is_initial=True,
            )
        ],
        strict=False,
    )

    assert any(
        issue.location == "check.complex"
        and "incomplete phase graph" in issue.message
        and "flag_b.blocked" in issue.message
        and "flag_b is not assigned to any phase" in issue.message
        for issue in system.compile_report.issues
    )
    assert not any(
        "flag_a.ready" in issue.message or "flag_c.level" in issue.message
        for issue in system.compile_report.issues
    )


def test_elseif_guard_source_participates_in_graph_completeness() -> None:
    flag_a = Flag(name="flag_a")
    flag_b = Flag(name="flag_b")

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "check",
                nodes=(flag_a,),
                transitions=(
                    If(V(flag_a.State.ready), "first", name="first"),
                    ElseIf(V(flag_b.State.ready), "second", name="second"),
                    Else(terminate, name="fallback"),
                ),
                is_initial=True,
            ),
            Phase("first", nodes=(), transitions=(Goto(terminate),)),
            Phase("second", nodes=(), transitions=(Goto(terminate),)),
        ],
        strict=False,
    )

    assert any(
        issue.location == "check.second"
        and "incomplete phase graph" in issue.message
        and "flag_b.ready" in issue.message
        for issue in system.compile_report.issues
    )


def test_else_does_not_add_guard_sources() -> None:
    flag = Flag()

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "check",
                nodes=(flag,),
                transitions=(
                    If(V(flag.State.ready), "next", name="ready"),
                    Else(terminate, name="fallback"),
                ),
                is_initial=True,
            ),
            Phase("next", nodes=(), transitions=(Goto(terminate),)),
        ],
    )

    assert system.compile_report.ok


def test_goto_does_not_add_guard_sources() -> None:
    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "start",
                nodes=(Worker(),),
                transitions=(Goto("next"),),
                is_initial=True,
            ),
            Phase("next", nodes=(), transitions=(Goto(terminate),)),
        ],
    )

    assert system.compile_report.ok


def test_python_lambda_guard_does_not_participate_in_source_graph() -> None:
    flag = Flag(name="external_flag")

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "check",
                nodes=(Worker(),),
                transitions=(
                    If(
                        lambda state: bool(state.get("external_flag.ready", False)),
                        terminate,
                        name="lambda",
                    ),
                ),
                is_initial=True,
            )
        ],
        strict=False,
    )

    assert not any("external_flag" in issue.message for issue in system.compile_report.issues)
    assert not any("external_flag" in issue.message for issue in system.compile_report.warnings)
    assert flag.node_id == "external_flag"


def test_elif_is_identical_to_elseif_transition_constructor() -> None:
    transition = Elif(V(Flag.State.ready), terminate)
    elseif_transition = ElseIf(V(Flag.State.ready), terminate)

    assert transition.kind == elseif_transition.kind
    assert transition.name == elseif_transition.name
    assert transition.target is elseif_transition.target
    assert getattr(transition.predicate, "variables") == getattr(
        elseif_transition.predicate,
        "variables",
    )
    assert transition.kind == "elseif"
    assert transition.name == "elseif"


def test_elif_participates_in_if_chains_like_elseif() -> None:
    mode = Flag(name="mode", ready=False, level=2)
    target = Worker()

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "route",
                nodes=(mode,),
                transitions=(
                    If(V(mode.State.ready), "ready", name="ready"),
                    Elif(V(mode.State.level) == 2, "level-two", name="level-two"),
                    Else(terminate, name="fallback"),
                ),
                is_initial=True,
            ),
            Phase("ready", nodes=(), transitions=(Goto(terminate),)),
            Phase("level-two", nodes=(target,), transitions=(Goto(terminate),)),
        ],
    )

    assert system.compile_report.ok
    system.run(steps=1)
    assert [record.phase for record in system.history][-2:] == ["route", "level-two"]
    assert system.snapshot()["Worker.ran"] is True


def test_elif_can_be_mixed_with_elseif_in_the_same_chain() -> None:
    mode = Flag(name="mode", ready=False, blocked=False, level=3)
    target = Worker()

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "route",
                nodes=(mode,),
                transitions=(
                    If(V(mode.State.ready), "ready", name="ready"),
                    Elif(V(mode.State.blocked), "blocked", name="blocked"),
                    ElseIf(V(mode.State.level) == 3, "level-three", name="level-three"),
                    Else(terminate, name="fallback"),
                ),
                is_initial=True,
            ),
            Phase("ready", nodes=(), transitions=(Goto(terminate),)),
            Phase("blocked", nodes=(), transitions=(Goto(terminate),)),
            Phase("level-three", nodes=(target,), transitions=(Goto(terminate),)),
        ],
    )

    assert system.compile_report.ok
    system.run(steps=1)
    assert [record.phase for record in system.history][-2:] == ["route", "level-three"]


def test_elif_requires_open_if_chain_like_elseif() -> None:
    with pytest.raises(CompileError) as exc_info:
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "bad",
                    nodes=(Flag(),),
                    transitions=(
                        Elif(V(Flag.State.ready), terminate),
                        Else(terminate),
                    ),
                    is_initial=True,
                )
            ],
        )

    assert any(
        issue.location == "bad.elseif" and issue.message == "ElseIf must follow If or ElseIf"
        for issue in exc_info.value.report.issues
    )


def test_elif_after_else_is_compile_error_like_elseif() -> None:
    with pytest.raises(CompileError) as exc_info:
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "bad",
                    nodes=(Flag(),),
                    transitions=(
                        If(V(Flag.State.ready), terminate),
                        Else(terminate),
                        Elif(~V(Flag.State.ready), terminate),
                    ),
                    is_initial=True,
                )
            ],
        )

    assert any(
        issue.location == "bad.elseif" and issue.message == "ElseIf must follow If or ElseIf"
        for issue in exc_info.value.report.issues
    )


def test_multi_phase_pipeline_resolves_instance_inputs_and_guards_across_branches() -> None:
    class Sensor(Node):
        class State(NodeState):
            value: int = Var(init=0)
            ready: bool = Var(init=True)

        def __init__(self, value: int, *, name: str) -> None:
            super().__init__(name=name)
            self.value = value

        def update(self) -> State:
            return self.State(value=self.value, ready=True)

    class Controller(Node):
        class Inputs(NodeInputs):
            value: int = Input()

        class State(NodeState):
            command: int = Var()
            high: bool = Var(init=True)

        def update(self, inputs: Inputs) -> State:
            return self.State(command=inputs.value + 10, high=inputs.value > 5)

    class Actuator(Node):
        class Inputs(NodeInputs):
            command: int = Input()

        class State(NodeState):
            applied: int = Var(init=0)

        def update(self, inputs: Inputs) -> State:
            return self.State(applied=inputs.command)

    primary = Sensor(7, name="primary")
    backup = Sensor(2, name="backup")
    controller = Controller(name="controller")
    actuator = Actuator(name="actuator")
    port(controller.Inputs.value).connect(primary.State.value)
    port(actuator.Inputs.command).connect(controller.State.command)

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "sense",
                nodes=(primary, backup),
                transitions=(
                    If(V(primary.State.ready), "control", name="primary-ready"),
                    ElseIf(V(backup.State.ready), "control", name="backup-ready"),
                    Else(terminate, name="no-sensor"),
                ),
                is_initial=True,
            ),
            Phase(
                "control",
                nodes=(controller,),
                transitions=(
                    If(V(controller.State.high), "actuate", name="high"),
                    Else(terminate, name="low"),
                ),
            ),
            Phase("actuate", nodes=(actuator,), transitions=(Goto(terminate),)),
        ],
    )

    assert system.compile_report.ok
    assert system.compile_report.inputs == {
        "controller.value": "primary.value",
        "actuator.command": "controller.command",
    }
    assert system.compile_report.phase_schedules["control"] == ("controller",)
    system.run(steps=1)
    assert [record.phase for record in system.history][-3:] == [
        "sense",
        "control",
        "actuate",
    ]
    assert system.snapshot()["actuator.applied"] == 17


def test_complex_phase_graph_reports_all_missing_instance_guard_and_input_sources() -> None:
    class ExternalConfig(Node):
        class State(NodeState):
            enabled: bool = Var(init=True)
            gain: int = Var(init=2)

    class ExternalMode(Node):
        class State(NodeState):
            armed: bool = Var(init=True)

    class Processor(Node):
        class Inputs(NodeInputs):
            gain: int = Input()

        class State(NodeState):
            score: int = Var(init=0)

        def update(self, inputs: Inputs) -> State:
            return self.State(score=inputs.gain)

    config = ExternalConfig(name="config")
    mode = ExternalMode(name="mode")
    processor = Processor(name="processor")
    port(processor.Inputs.gain).connect(config.State.gain)

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "process",
                nodes=(processor,),
                transitions=(
                    If(
                        V(config.State.enabled) & V(mode.State.armed),
                        terminate,
                        name="external-ready",
                    ),
                    Else(terminate, name="fallback"),
                ),
                is_initial=True,
            )
        ],
        strict=False,
    )

    assert any(
        issue.location == "processor.gain"
        and "incomplete phase graph" in issue.message
        and "config.gain" in issue.message
        for issue in system.compile_report.issues
    )
    assert any(
        issue.location == "process.external-ready"
        and "incomplete phase graph" in issue.message
        and "config.enabled" in issue.message
        for issue in system.compile_report.issues
    )
    assert any(
        issue.location == "process.external-ready"
        and "incomplete phase graph" in issue.message
        and "mode.armed" in issue.message
        for issue in system.compile_report.issues
    )
    assert not any(
        "unknown input source" in issue.message or "unknown guard variable" in issue.message
        for issue in system.compile_report.issues
    )


def test_class_level_guard_is_ambiguous_even_when_duplicate_instances_live_in_different_branches() -> (
    None
):
    class Mode(Node):
        class State(NodeState):
            ready: bool = Var(init=True)

    class Work(Node):
        class State(NodeState):
            done: bool = Var(init=False)

    with pytest.raises(CompileError) as exc_info:
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "left",
                    nodes=(Mode(name="left_mode"),),
                    transitions=(Goto("right"),),
                    is_initial=True,
                ),
                Phase(
                    "right",
                    nodes=(Mode(name="right_mode"),),
                    transitions=(Goto("join"),),
                ),
                Phase(
                    "join",
                    nodes=(Work(),),
                    transitions=(
                        If(V(Mode.State.ready), terminate, name="ready"),
                        Else(terminate, name="not-ready"),
                    ),
                ),
            ],
        )

    assert any(
        issue.location == "join.ready"
        and "ambiguous guard variable 'Mode.ready'" in issue.message
        and "left_mode.ready" in issue.message
        and "right_mode.ready" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_instance_guard_keeps_branch_join_unambiguous_with_duplicate_node_classes() -> None:
    class Mode(Node):
        class State(NodeState):
            ready: bool = Var(init=True)

        def update(self) -> State:
            return self.State(ready=True)

    class Work(Node):
        class State(NodeState):
            done: bool = Var(init=False)

        def update(self) -> State:
            return self.State(done=True)

    left_mode = Mode(name="left_mode")
    right_mode = Mode(name="right_mode")
    work = Work()

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "left",
                nodes=(left_mode,),
                transitions=(Goto("right"),),
                is_initial=True,
            ),
            Phase("right", nodes=(right_mode,), transitions=(Goto("join"),)),
            Phase(
                "join",
                nodes=(work,),
                transitions=(
                    If(V(right_mode.State.ready), terminate, name="ready"),
                    Else(terminate, name="not-ready"),
                ),
            ),
        ],
    )

    assert system.compile_report.ok
    system.run(steps=1)
    assert [record.phase for record in system.history] == ["left", "right", "join"]
    assert system.snapshot()["Work.done"] is True


def test_nested_elseif_chains_collect_sources_from_every_chain_segment() -> None:
    gate_a = Flag(name="gate_a")
    gate_b = Flag(name="gate_b")
    missing_gate = Flag(name="missing_gate")
    gate_c = Flag(name="gate_c")

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "route",
                nodes=(gate_a, gate_b, gate_c),
                transitions=(
                    If(V(gate_a.State.ready), "a", name="a"),
                    ElseIf(V(gate_b.State.ready), "b", name="b"),
                    Else(terminate, name="fallback"),
                    If(V(missing_gate.State.ready), "missing", name="missing"),
                    ElseIf(V(gate_c.State.ready), "c", name="c"),
                    Else(terminate, name="second-fallback"),
                ),
                is_initial=True,
            ),
            Phase("a", nodes=(), transitions=(Goto(terminate),)),
            Phase("b", nodes=(), transitions=(Goto(terminate),)),
            Phase("c", nodes=(), transitions=(Goto(terminate),)),
            Phase("missing", nodes=(), transitions=(Goto(terminate),)),
        ],
        strict=False,
    )

    assert any(
        issue.location == "route.missing"
        and "incomplete phase graph" in issue.message
        and "missing_gate.ready" in issue.message
        for issue in system.compile_report.issues
    )
    assert not any(
        "gate_a.ready" in issue.message
        or "gate_b.ready" in issue.message
        or "gate_c.ready" in issue.message
        for issue in system.compile_report.issues
    )
    assert any(
        issue.location == "route.missing" and issue.message.startswith("transition follows Else")
        for issue in system.compile_report.warnings
    )
