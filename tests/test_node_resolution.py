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
    NodeOutputs,
    Output,
    Phase,
    PhasedReactiveSystem,
    V,
    port,
    terminate,
)


class Source(Node):
    class Outputs(NodeOutputs):
        value: int = Output(initial=1)
        ready: bool = Output(initial=True)

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

    def run(self) -> Outputs:
        return self.Outputs(value=self.value, ready=self.ready)


class Sink(Node):
    class Inputs(NodeInputs):
        value: int = Input(source=Source.Outputs.value)

    class Outputs(NodeOutputs):
        seen: int = Output(initial=0)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(seen=inputs.value)


class UnconnectedSink(Node):
    class Inputs(NodeInputs):
        value: int = Input()

    class Outputs(NodeOutputs):
        seen: int = Output(initial=0)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(seen=inputs.value)


class Worker(Node):
    class Outputs(NodeOutputs):
        ran: bool = Output(initial=False)

    def run(self) -> Outputs:
        return self.Outputs(ran=True)


class Flag(Node):
    class Outputs(NodeOutputs):
        ready: bool = Output(initial=True)
        blocked: bool = Output(initial=False)
        level: int = Output(initial=0)

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

    def run(self) -> Outputs:
        return self.Outputs(
            ready=self.ready,
            blocked=self.blocked,
            level=self.level,
        )


class LinkA(Node):
    class Outputs(NodeOutputs):
        value: int = Output(initial=1)


class LinkB(Node):
    class Inputs(NodeInputs):
        value: int = Input(source=LinkA.Outputs.value)

    class Outputs(NodeOutputs):
        value: int = Output(initial=0)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(value=inputs.value)


class LinkC(Node):
    class Inputs(NodeInputs):
        value: int = Input(source=LinkB.Outputs.value)

    class Outputs(NodeOutputs):
        value: int = Output(initial=0)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(value=inputs.value)


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
        "ambiguous input source 'Source.value'" in message
        and "use instance connection" in message
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
    port(sink.Inputs.value).connect(source_b.Outputs.value)

    system = _single_phase_system(source_a, source_b, sink)

    assert system.compile_report.ok
    assert system.compile_report.inputs["UnconnectedSink.value"] == "Source_2.value"
    system.step()
    assert system.snapshot()["UnconnectedSink.seen"] == 2


def test_instance_bound_input_ref_may_point_to_node_in_another_phase() -> None:
    source = Source(value=3)
    sink = UnconnectedSink()
    port(sink.Inputs.value).connect(source.Outputs.value)

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
    port(sink.Inputs.value).connect(source.Outputs.value)

    system = _single_phase_system(sink, strict=False)

    assert any(
        issue.location == "UnconnectedSink.value"
        and "incomplete phase graph" in issue.message
        and "Source.value" in issue.message
        and "Source is not assigned to any phase" in issue.message
        for issue in system.compile_report.issues
    )
    assert not any(
        issue.location == "UnconnectedSink.value"
        and "unknown input source" in issue.message
        for issue in system.compile_report.issues
    )


def test_input_closure_is_not_auto_completed_from_missing_producers() -> None:
    system = _single_phase_system(LinkC(), strict=False)

    assert any(
        issue.location == "LinkC.value"
        and "unknown input source 'LinkB.value'" in issue.message
        for issue in system.compile_report.issues
    )
    assert tuple(system.compile_report.nodes) == ("LinkC",)


def test_input_closure_checks_second_hop_when_intermediate_node_is_covered() -> None:
    system = _single_phase_system(LinkB(), LinkC(), strict=False)

    assert any(
        issue.location == "LinkB.value"
        and "unknown input source 'LinkA.value'" in issue.message
        for issue in system.compile_report.issues
    )
    assert not any(
        issue.location == "LinkC.value" for issue in system.compile_report.issues
    )


def test_phase_nodes_accept_instances_only() -> None:
    with pytest.raises(TypeError, match="Phase.nodes accepts node instances only"):
        Phase(
            "bad",
            nodes=(Source,),  # pyright: ignore[reportArgumentType]
            transitions=(Goto(terminate),),
            is_initial=True,
        )


def test_duplicate_implicit_names_do_not_break_instance_input_refs() -> None:
    source_a = Source(value=1)
    source_b = Source(value=2)
    sink = UnconnectedSink()
    port(sink.Inputs.value).connect(source_b.Outputs.value)

    system = _single_phase_system(source_a, source_b, sink)

    assert system.compile_report.nodes == ("Source", "Source_2", "UnconnectedSink")
    assert system.compile_report.inputs["UnconnectedSink.value"] == "Source_2.value"


def test_duplicate_explicit_names_are_rejected() -> None:
    with pytest.raises(CompileError) as exc_info:
        _single_phase_system(Source(name="source"), Source(name="source"))

    assert any(
        issue.location == "source"
        and issue.message == "node name is declared more than once"
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
                    If(V(flag.Outputs.ready), terminate, name="ready"),
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
                    If(V(flag.Outputs.ready), terminate, name="ready"),
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
                    If(V(flag.Outputs.ready), terminate, name="ready"),
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
        issue.location == "work.ready"
        and "unknown guard variable" in issue.message
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
                    If(V(Flag.Outputs.ready), terminate, name="ready"),
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
                    If(V(Flag.Outputs.ready), terminate, name="ready"),
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
                        If(V(Flag.Outputs.ready), terminate, name="ready"),
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
        and "use instance output reference" in issue.message
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
                        If(V(Flag.Outputs.ready), terminate, name="ready"),
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
                    If(V(flag_b.Outputs.ready), terminate, name="ready"),
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
                    If(V(flag_b.Outputs.ready), terminate, name="ready"),
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
                        (
                            V(flag_a.Outputs.ready)
                            & ~V(flag_b.Outputs.blocked)
                        )
                        | (V(flag_c.Outputs.level) > 3),
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
                        (
                            V(flag_a.Outputs.ready)
                            & ~V(flag_b.Outputs.blocked)
                        )
                        | (V(flag_c.Outputs.level) > 3),
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
                    If(V(flag_a.Outputs.ready), "first", name="first"),
                    ElseIf(V(flag_b.Outputs.ready), "second", name="second"),
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
                    If(V(flag.Outputs.ready), "next", name="ready"),
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

    assert not any(
        "external_flag" in issue.message for issue in system.compile_report.issues
    )
    assert not any(
        "external_flag" in issue.message for issue in system.compile_report.warnings
    )
    assert flag.node_id == "external_flag"


def test_elif_is_identical_to_elseif_transition_constructor() -> None:
    transition = Elif(V(Flag.Outputs.ready), terminate)
    elseif_transition = ElseIf(V(Flag.Outputs.ready), terminate)

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
                    If(V(mode.Outputs.ready), "ready", name="ready"),
                    Elif(V(mode.Outputs.level) == 2, "level-two", name="level-two"),
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
                    If(V(mode.Outputs.ready), "ready", name="ready"),
                    Elif(V(mode.Outputs.blocked), "blocked", name="blocked"),
                    ElseIf(V(mode.Outputs.level) == 3, "level-three", name="level-three"),
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
                        Elif(V(Flag.Outputs.ready), terminate),
                        Else(terminate),
                    ),
                    is_initial=True,
                )
            ],
        )

    assert any(
        issue.location == "bad.elseif"
        and issue.message == "ElseIf must follow If or ElseIf"
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
                        If(V(Flag.Outputs.ready), terminate),
                        Else(terminate),
                        Elif(~V(Flag.Outputs.ready), terminate),
                    ),
                    is_initial=True,
                )
            ],
        )

    assert any(
        issue.location == "bad.elseif"
        and issue.message == "ElseIf must follow If or ElseIf"
        for issue in exc_info.value.report.issues
    )


def test_multi_phase_pipeline_resolves_instance_inputs_and_guards_across_branches() -> None:
    class Sensor(Node):
        class Outputs(NodeOutputs):
            value: int = Output(initial=0)
            ready: bool = Output(initial=True)

        def __init__(self, value: int, *, name: str) -> None:
            super().__init__(name=name)
            self.value = value

        def run(self) -> Outputs:
            return self.Outputs(value=self.value, ready=True)

    class Controller(Node):
        class Inputs(NodeInputs):
            value: int = Input()

        class Outputs(NodeOutputs):
            command: int = Output()
            high: bool = Output(initial=True)

        def run(self, inputs: Inputs) -> Outputs:
            return self.Outputs(command=inputs.value + 10, high=inputs.value > 5)

    class Actuator(Node):
        class Inputs(NodeInputs):
            command: int = Input()

        class Outputs(NodeOutputs):
            applied: int = Output(initial=0)

        def run(self, inputs: Inputs) -> Outputs:
            return self.Outputs(applied=inputs.command)

    primary = Sensor(7, name="primary")
    backup = Sensor(2, name="backup")
    controller = Controller(name="controller")
    actuator = Actuator(name="actuator")
    port(controller.Inputs.value).connect(primary.Outputs.value)
    port(actuator.Inputs.command).connect(controller.Outputs.command)

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "sense",
                nodes=(primary, backup),
                transitions=(
                    If(V(primary.Outputs.ready), "control", name="primary-ready"),
                    ElseIf(V(backup.Outputs.ready), "control", name="backup-ready"),
                    Else(terminate, name="no-sensor"),
                ),
                is_initial=True,
            ),
            Phase(
                "control",
                nodes=(controller,),
                transitions=(
                    If(V(controller.Outputs.high), "actuate", name="high"),
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
        class Outputs(NodeOutputs):
            enabled: bool = Output(initial=True)
            gain: int = Output(initial=2)

    class ExternalMode(Node):
        class Outputs(NodeOutputs):
            armed: bool = Output(initial=True)

    class Processor(Node):
        class Inputs(NodeInputs):
            gain: int = Input()

        class Outputs(NodeOutputs):
            score: int = Output(initial=0)

        def run(self, inputs: Inputs) -> Outputs:
            return self.Outputs(score=inputs.gain)

    config = ExternalConfig(name="config")
    mode = ExternalMode(name="mode")
    processor = Processor(name="processor")
    port(processor.Inputs.gain).connect(config.Outputs.gain)

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "process",
                nodes=(processor,),
                transitions=(
                    If(
                        V(config.Outputs.enabled) & V(mode.Outputs.armed),
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
        "unknown input source" in issue.message
        or "unknown guard variable" in issue.message
        for issue in system.compile_report.issues
    )


def test_class_level_guard_is_ambiguous_even_when_duplicate_instances_live_in_different_branches() -> None:
    class Mode(Node):
        class Outputs(NodeOutputs):
            ready: bool = Output(initial=True)

    class Work(Node):
        class Outputs(NodeOutputs):
            done: bool = Output(initial=False)

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
                        If(V(Mode.Outputs.ready), terminate, name="ready"),
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
        class Outputs(NodeOutputs):
            ready: bool = Output(initial=True)

        def run(self) -> Outputs:
            return self.Outputs(ready=True)

    class Work(Node):
        class Outputs(NodeOutputs):
            done: bool = Output(initial=False)

        def run(self) -> Outputs:
            return self.Outputs(done=True)

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
                    If(V(right_mode.Outputs.ready), terminate, name="ready"),
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
                    If(V(gate_a.Outputs.ready), "a", name="a"),
                    ElseIf(V(gate_b.Outputs.ready), "b", name="b"),
                    Else(terminate, name="fallback"),
                    If(V(missing_gate.Outputs.ready), "missing", name="missing"),
                    ElseIf(V(gate_c.Outputs.ready), "c", name="c"),
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
        issue.location == "route.missing"
        and issue.message.startswith("transition follows Else")
        for issue in system.compile_report.warnings
    )
