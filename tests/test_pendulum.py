from enum import Enum
from typing import Any, cast

import pytest

from regelum import (
    CompileError,
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
    VarPort,
    port,
    terminate,
)
from regelum.examples.c1_violation import build_system as build_c1_violation_system
from regelum.examples.c2star_split_writes import (
    build_system as build_c2star_split_writes_system,
)
from regelum.examples.c3_violation import build_system as build_c3_violation_system
from regelum.examples.complex_c3_partition import (
    build_bad_overlap_system,
)
from regelum.examples.complex_c3_partition import (
    build_ok_system as build_c3_ok_system,
)
from regelum.examples.complex_safety_loop import (
    build_bad_c1_system,
)
from regelum.examples.complex_safety_loop import (
    build_ok_system as build_complex_ok_system,
)
from regelum.examples.mohanty2023_smart_home import (
    LS1,
    S2,
    SS0,
    build_fig4_system,
)
from regelum.examples.pendulum import build_system
from regelum.examples.safe_recovery_pipeline import (
    build_system as build_safe_recovery_system,
)
from regelum.examples.smart_home_async_world import (
    build_async_world,
    build_async_world_with_cycle,
)
from regelum.examples.smart_home_safety_journey import (
    ACT_FIRE_OVER_LEAK,
)
from regelum.examples.smart_home_safety_journey import (
    build_v1 as build_smart_home_v1,
)
from regelum.examples.smart_home_safety_journey import (
    build_v2 as build_smart_home_v2,
)
from regelum.examples.smart_home_safety_journey import (
    build_v3 as build_smart_home_v3,
)
from regelum.examples.smart_home_safety_journey import (
    build_v4 as build_smart_home_v4,
)


def _tick_system(
    nodes: list[Node],
    **kwargs: Any,
) -> PhasedReactiveSystem:
    return PhasedReactiveSystem(
        phases=[
            Phase(
                "tick",
                nodes=tuple(nodes),
                transitions=(Goto(terminate),),
                is_initial=True,
            )
        ],
        **kwargs,
    )


def test_pendulum_compiles() -> None:
    system = build_system()

    report = system.compile_report

    assert report.ok
    assert report.nodes == ("PDController", "Logger", "PendulumPlant")
    assert report.inputs["PDController.theta"] == "PendulumPlant.theta"
    assert report.inputs["PendulumPlant.torque"] == "PDController.torque"
    assert report.inputs["Logger.time"] == "PendulumPlant.time"
    assert report.inputs["Logger.samples"] == "Logger.samples"
    assert report.unlinked_inputs == ()
    assert set(report.linked_inputs) == set(report.inputs)
    assert report.phase_schedules["control"] == ("PDController", "Logger")
    assert report.phase_schedules["plant"] == ("PendulumPlant",)
    assert report.phase_dependency_edges["control"] == (("PDController", "Logger"),)
    assert "PDController.torque" not in report.minimal_initial_state_vars
    assert "PendulumPlant.theta" in report.minimal_initial_state_vars


def test_pendulum_runs_and_logs_samples() -> None:
    system = build_system()

    system.run(steps=5)

    snapshot = system.snapshot()
    assert [record.phase for record in system.history[:3]] == [
        "control",
        "control",
        "plant",
    ]
    assert len(snapshot["Logger.samples"]) == 5
    assert snapshot["PendulumPlant.time"] == pytest.approx(10.1)
    assert snapshot["Logger.samples"][-1][0] == pytest.approx(10.08)
    assert snapshot["PendulumPlant.theta"] != 0.35
    assert abs(snapshot["PDController.torque"]) <= 10.0


def test_compile_rejects_unknown_input_source() -> None:
    class Broken(Node):
        class Inputs(NodeInputs):
            value: float = Input("Missing.State.value")

        class State(NodeState):
            result: float = Var(init=0.0)

    with pytest.raises(CompileError) as exc_info:
        _tick_system([Broken()])

    assert not exc_info.value.report.ok
    assert exc_info.value.report.issues[0].location == "Broken.value"


def test_compile_rejects_unconnected_input() -> None:
    class Broken(Node):
        class Inputs(NodeInputs):
            value: float = Input()

        class State(NodeState):
            result: float = Var(init=0.0)

    with pytest.raises(CompileError) as exc_info:
        _tick_system([Broken()])

    assert not exc_info.value.report.ok
    assert exc_info.value.report.issues[0].location == "Broken.value"


def test_compile_rejects_duplicate_state_paths() -> None:
    class Source(Node):
        class State(NodeState):
            value: float = Var(init=0.0)

    with pytest.raises(CompileError) as exc_info:
        _tick_system([Source(name="source"), Source(name="source")])

    assert not exc_info.value.report.ok
    assert exc_info.value.report.issues[0].location == "source.value"


def test_compile_rejects_missing_state_initial_value_before_first_read() -> None:
    class Source(Node):
        class Inputs(NodeInputs):
            previous: float = Input(src=lambda: Source.State.value)

        class State(NodeState):
            value: float = Var()

        def update(self, inputs: Inputs) -> State:
            return self.State(value=inputs.previous + 1)

    with pytest.raises(CompileError) as exc_info:
        _tick_system([Source()])

    assert not exc_info.value.report.ok
    assert exc_info.value.report.issues[0].location == "Source.value"
    assert exc_info.value.report.state_vars_without_initial == ("Source.value",)
    assert exc_info.value.report.required_initial_state_vars == {"Source.value": ("Source.previous",)}
    assert (
        exc_info.value.report.issues[0].message
        == "state variable initial value is required before first read by ('Source.previous',)"
    )


def test_state_var_without_initial_is_allowed_when_written_before_read() -> None:
    class Source(Node):
        class State(NodeState):
            value: int = Var()

        def update(self) -> State:
            return self.State(value=5)

    class Sink(Node):
        class Inputs(NodeInputs):
            value: int = Input(src=lambda: Source.State.value)

        class State(NodeState):
            seen: int = Var(init=0)

        def update(self, inputs: Inputs) -> State:
            return self.State(seen=inputs.value)

    source = Source()
    sink = Sink()
    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "copy",
                nodes=(sink, source),
                transitions=(Goto(terminate),),
                is_initial=True,
            ),
        ],
    )

    assert "Source.value" not in system.snapshot()
    assert system.compile_report.state_vars_without_initial == ("Source.value",)
    assert "Source.value" not in system.compile_report.minimal_initial_state_vars
    assert system.compile_report.required_initial_state_vars == {}
    assert system.compile_report.phase_schedules["copy"] == ("Source", "Sink")
    assert system.compile_report.phase_dependency_edges["copy"] == (("Source", "Sink"),)

    system.step()

    assert system.snapshot()["Source.value"] == 5
    assert system.snapshot()["Sink.seen"] == 5


def test_system_can_be_created_with_compile_issues_in_non_strict_mode() -> None:
    class Broken(Node):
        class Inputs(NodeInputs):
            value: int

        class State(NodeState):
            seen: int

        def update(self, inputs: Inputs) -> State:
            return self.State(seen=inputs.value)

    system = _tick_system([Broken()], strict=False)

    assert not system.compile_report.ok
    assert system.compile_report.unlinked_inputs == ("Broken.value",)
    assert system.compile_report.minimal_initial_state_vars == ()
    assert system.compile_report.state_vars_without_initial == ("Broken.seen",)


def test_initial_state_overrides_default_state_initial_values() -> None:
    class Source(Node):
        class State(NodeState):
            value: int = Var(init=1)

        def update(self) -> State:
            return self.State(value=5)

    system = _tick_system(
        [Source()],
        initial_state={Source.State.value: 3},
    )

    assert system.snapshot()["Source.value"] == 3

    system.step()

    assert system.snapshot()["Source.value"] == 5

    system.reset(initial_state={Source.State.value: 7})

    assert system.snapshot()["Source.value"] == 7


def test_initial_state_can_supply_required_state_var_without_initial() -> None:
    class Source(Node):
        class Inputs(NodeInputs):
            previous: int = Input(src=lambda: Source.State.value)

        class State(NodeState):
            value: int

        def update(self, inputs: Inputs) -> State:
            return self.State(value=inputs.previous + 1)

    system = _tick_system(
        [Source()],
        initial_state={Source.State.value: 10},
    )

    assert system.compile_report.ok
    assert system.compile_report.required_initial_state_vars == {"Source.value": ("Source.previous",)}
    assert system.snapshot()["Source.value"] == 10

    system.step()

    assert system.snapshot()["Source.value"] == 11


def test_bare_annotations_install_default_input_and_state_ports() -> None:
    class Source(Node):
        class State(NodeState):
            value: int

        def update(self) -> State:
            return self.State(value=5)

    class Sink(Node):
        class Inputs(NodeInputs):
            value: int

        class State(NodeState):
            seen: int

        def update(self, inputs: Inputs) -> State:
            return self.State(seen=inputs.value)

    source = Source()
    sink = Sink()
    port(sink.Inputs.value).connect(source.State.value)
    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "copy",
                nodes=(sink, source),
                transitions=(Goto(terminate),),
                is_initial=True,
            )
        ],
    )

    assert system.compile_report.ok

    system.step()

    assert system.compile_report.inputs["Sink.value"] == "Source.value"
    assert system.snapshot()["Sink.seen"] == 5


def test_update_parameters_can_declare_inputs() -> None:
    class Source(Node):
        class State(NodeState):
            value: int

        def update(self) -> State:
            return self.State(value=5)

    class Sink(Node):
        class State(NodeState):
            seen: int

        def update(
            self,
            value: int = Input(src=lambda: Source.State.value),
        ) -> State:
            return self.State(seen=value)

    source = Source()
    sink = Sink()
    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "copy",
                nodes=(sink, source),
                transitions=(Goto(terminate),),
                is_initial=True,
            )
        ],
    )

    system.step()

    assert system.compile_report.inputs["Sink.value"] == "Source.value"
    assert system.compile_report.phase_schedules["copy"] == ("Source", "Sink")
    assert system.snapshot()["Sink.seen"] == 5


def test_update_parameter_inputs_cannot_be_mixed_with_inputs_class() -> None:
    class Source(Node):
        class State(NodeState):
            value: int = Var(init=1)

    class Sink(Node):
        class Inputs(NodeInputs):
            other: int = Input(src=lambda: Source.State.value)

        class State(NodeState):
            seen: int = Var(init=0)

        def update(
            self,
            value: int = Input(src=lambda: Source.State.value),
        ) -> State:
            return self.State(seen=value)

    with pytest.raises(CompileError) as exc_info:
        _tick_system([Source(), Sink()])

    assert any(
        issue.location == "Sink.update"
        and issue.message
        == "define inputs either as a NodeInputs namespace or as update(...) parameters, not both"
        for issue in exc_info.value.report.issues
    )


def test_node_can_use_custom_input_and_state_namespace_names() -> None:
    class Source(Node):
        class State(NodeState):
            value: int = Var(init=1)

        def update(self) -> State:
            return self.State(value=5)

    class Sink(Node):
        class In(NodeInputs):
            value: int = Input(src=lambda: Source.State.value)

        class State(NodeState):
            seen: int = Var()

        def update(self, inputs: In) -> State:
            return self.State(seen=inputs.value)

    source = Source()
    sink = Sink()
    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "tick",
                nodes=(sink, source),
                transitions=(Goto(terminate),),
                is_initial=True,
            )
        ],
    )

    assert hasattr(Source.State, "value")
    assert system.compile_report.phase_schedules["tick"] == ("Source", "Sink")

    system.step()

    assert system.snapshot()["Source.value"] == 5
    assert system.snapshot()["Sink.seen"] == 5


def test_instance_bound_ports_use_state_namespace() -> None:
    class Source(Node):
        class State(NodeState):
            value: int = Var(init=0)

    class Sink(Node):
        class In(NodeInputs):
            value: int = Input()

        class State(NodeState):
            seen: int = Var(init=0)

    source = Source()
    sink = Sink()
    connection = port(sink.In.value).connect(source.State.value)

    assert connection.input.path == "Sink.value"
    assert connection.source.path == "Source.value"
    assert source.State.value.path == "Source.value"


def test_string_source_paths_support_custom_state_namespace_names() -> None:
    class Source(Node):
        class State(NodeState):
            value: int = Var(init=1)

        def update(self) -> State:
            return self.State(value=8)

    class Sink(Node):
        class In(NodeInputs):
            value: int = Input(src="Source.State.value")

        class State(NodeState):
            seen: int = Var()

        def update(self, inputs: In) -> State:
            return self.State(seen=inputs.value)

    source = Source()
    sink = Sink()
    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "tick",
                nodes=(source, sink),
                transitions=(Goto(terminate),),
                is_initial=True,
            )
        ],
    )

    assert system.compile_report.inputs["Sink.value"] == "Source.value"
    system.step()
    assert system.snapshot()["Sink.seen"] == 8


def test_initial_state_supports_custom_namespace_class_and_instance_ports() -> None:
    class Source(Node):
        class State(NodeState):
            value: int = Var(init=1)

    source = Source(name="source")
    class_system = _tick_system([Source()], initial_state={Source.State.value: 5})
    instance_system = _tick_system([source], initial_state={source.State.value: 7})

    assert class_system.snapshot()["Source.value"] == 5
    assert instance_system.snapshot()["source.value"] == 7


def test_guards_support_custom_namespace_state_refs() -> None:
    class Source(Node):
        class State(NodeState):
            value: int = Var(init=2)

        def update(self) -> State:
            return self.State(value=2)

    class Sink(Node):
        class State(NodeState):
            reached: bool = Var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    source = Source()
    sink = Sink()
    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "select",
                nodes=(source,),
                transitions=(
                    If(V(Source.State.value) > 1, "sink"),
                    Else(terminate),
                ),
                is_initial=True,
            ),
            Phase("sink", nodes=(sink,), transitions=(Goto(terminate),)),
        ],
    )

    system.step()

    assert system.snapshot()["Sink.reached"] is True


def test_instance_input_source_must_be_assigned_to_a_phase() -> None:
    class Source(Node):
        class State(NodeState):
            value: int = Var(init=1)

    class Sink(Node):
        class In(NodeInputs):
            value: int = Input()

        class State(NodeState):
            seen: int = Var(init=0)

    source = Source(name="source")
    sink = Sink(name="sink")
    port(sink.In.value).connect(source.State.value)

    with pytest.raises(CompileError) as exc_info:
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "tick",
                    nodes=(sink,),
                    transitions=(Goto(terminate),),
                    is_initial=True,
                )
            ]
        )

    assert any(
        issue.location == "sink.value"
        and "incomplete phase graph" in issue.message
        and "node source is not assigned to any phase" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_instance_guard_source_must_be_assigned_to_a_phase() -> None:
    class Source(Node):
        class State(NodeState):
            ready: bool = Var(init=True)

    source = Source(name="source")

    with pytest.raises(CompileError) as exc_info:
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "tick",
                    nodes=(),
                    transitions=(If(V(source.State.ready), terminate, name="ready"),),
                    is_initial=True,
                )
            ]
        )

    assert any(
        issue.location == "tick.ready"
        and "incomplete phase graph" in issue.message
        and "node source is not assigned to any phase" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_node_subclass_inherits_port_namespace_when_not_overridden() -> None:
    class Base(Node):
        class State(NodeState):
            value: int = Var(init=1)

    class Child(Base):
        pass

    system = _tick_system([Child()])

    assert hasattr(Child.State, "value")
    assert system.snapshot()["Child.value"] == 1


def test_node_subclass_can_override_port_namespace() -> None:
    class Base(Node):
        class State(NodeState):
            base_value: int = Var(init=1)

    class Child(Base):
        class State(NodeState):
            child_value: int = Var(init=2)

    system = _tick_system([Child()])

    assert system.compile_report.state_vars == ("Child.child_value",)
    assert system.snapshot()["Child.child_value"] == 2


def test_node_rejects_two_input_namespaces() -> None:
    with pytest.raises(TypeError, match="may define zero or one input namespace"):

        class Bad(Node):
            class In(NodeInputs):
                first: int = Input()

            class MoreIn(NodeInputs):
                second: int = Input()


def test_node_rejects_three_input_namespaces() -> None:
    with pytest.raises(TypeError, match="found 3: In, MoreIn, ExtraIn"):

        class Bad(Node):
            class In(NodeInputs):
                first: int = Input()

            class MoreIn(NodeInputs):
                second: int = Input()

            class ExtraIn(NodeInputs):
                third: int = Input()


def test_node_rejects_two_state_namespaces() -> None:
    with pytest.raises(TypeError, match="may define zero or one state namespace"):

        class Bad(Node):
            class State(NodeState):
                first: int = Var(init=1)

            class ExtraState(NodeState):
                second: int = Var(init=2)


def test_node_rejects_three_state_namespaces() -> None:
    with pytest.raises(TypeError, match="found 3: State, MoreState, ExtraState"):

        class Bad(Node):
            class State(NodeState):
                first: int = Var(init=1)

            class MoreState(NodeState):
                second: int = Var(init=2)

            class ExtraState(NodeState):
                third: int = Var(init=3)


def test_var_initial_accepts_zero_argument_callable() -> None:
    class Source(Node):
        class State(NodeState):
            value: int = Var(init=lambda: 7)

    system = _tick_system([Source()])

    assert system.snapshot()["Source.value"] == 7


def test_var_initial_accepts_node_argument_callable() -> None:
    class Source(Node):
        def __init__(self, value: int, *, name: str | None = None) -> None:
            super().__init__(name=name)
            self.value = value

        class State(NodeState):
            value: int = Var(init=lambda self: cast(Source, self).value)

    left = Source(3, name="left")
    right = Source(7, name="right")

    system = _tick_system([left, right])

    assert system.snapshot() == {
        "left.value": 3,
        "right.value": 7,
    }


def test_var_initial_rejects_callable_with_too_many_required_arguments() -> None:
    class Source(Node):
        class State(NodeState):
            value: int = Var(init=lambda first, second: 1)

    with pytest.raises(CompileError) as exc_info:
        _tick_system([Source()])

    assert any(
        issue.location == "Source.value"
        and "Var init callable must accept zero arguments or one node argument"
        in issue.message
        for issue in exc_info.value.report.issues
    )


def test_node_without_inputs_can_run_without_input_argument() -> None:
    class Source(Node):
        class State(NodeState):
            value: int = Var(init=lambda: 0)

        def update(self) -> State:
            return self.State(value=5)

    system = _tick_system([Source()])

    system.step()

    assert system.snapshot()["Source.value"] == 5


def test_state_var_can_be_enriched_by_reference() -> None:
    class Source(Node):
        class State(NodeState):
            doc: dict[str, Any] = Var(init=lambda: {})

        def update(self) -> State:
            return self.State(
                doc={
                    "request_id": "r-001",
                    "raw": {"text": "hello"},
                }
            )

    class Enricher(Node):
        class Inputs(NodeInputs):
            doc: dict[str, Any] = Input(src=lambda: Source.State.doc)

        class State(NodeState):
            doc: dict[str, Any] = Var(init=lambda: {})

        def update(self, inputs: Inputs) -> State:
            doc = inputs.doc
            raw = doc["raw"]
            assert isinstance(raw, dict)
            text = raw["text"]
            assert isinstance(text, str)
            doc["features"] = {"length": len(text)}
            return self.State(doc=doc)

    source = Source()
    enricher = Enricher()
    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "enrich",
                nodes=(source, enricher),
                transitions=(Goto(terminate),),
                is_initial=True,
            ),
        ],
    )

    source_doc_port = cast(VarPort[dict[str, Any]], Source.State.doc)
    enriched_doc_port = cast(VarPort[dict[str, Any]], Enricher.State.doc)

    system.step()

    source_doc = system.read(source_doc_port)
    enriched_doc = system.read(enriched_doc_port)
    assert source_doc is enriched_doc
    assert enriched_doc["features"] == {"length": 5}


def test_input_source_accepts_lazy_state_reference() -> None:
    class Early(Node):
        class Inputs(NodeInputs):
            value: int = Input(src=lambda: Later.State.value)
            previous: int = Input(src=lambda: Early.State.total)

        class State(NodeState):
            total: int = Var(init=0)

        def update(self, inputs: Inputs) -> State:
            return self.State(total=inputs.value + inputs.previous)

    class Later(Node):
        class State(NodeState):
            value: int = Var(init=2)

        def update(self, inputs: NodeInputs) -> State:
            return self.State(value=3)

    early = Early()
    later = Later()
    system = PhasedReactiveSystem(
        phases=[
            Phase("produce", nodes=(later,), transitions=(Goto("consume"),), is_initial=True),
            Phase("consume", nodes=(early,), transitions=(Goto(terminate),)),
        ],
    )

    assert system.compile_report.inputs["Early.value"] == "Later.value"
    assert system.compile_report.inputs["Early.previous"] == "Early.total"

    system.step()

    assert system.snapshot()["Early.total"] == 3


def test_phase_runs_nodes_in_topological_order() -> None:
    class Source(Node):
        class State(NodeState):
            value: int = Var(init=0)

        def update(self) -> State:
            return self.State(value=3)

    class Sink(Node):
        class Inputs(NodeInputs):
            value: int = Input(src=lambda: Source.State.value)

        class State(NodeState):
            seen: int = Var(init=0)

        def update(self, inputs: Inputs) -> State:
            return self.State(seen=inputs.value)

    source = Source()
    sink = Sink()
    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "copy",
                nodes=(sink, source),
                transitions=(Goto(terminate),),
                is_initial=True,
            )
        ],
    )

    system.step()

    assert [record.node for record in system.history] == ["Source", "Sink"]
    assert system.snapshot()["Sink.seen"] == 3


def test_guard_variable_accepts_state_reference() -> None:
    class Mode(Node):
        class State(NodeState):
            flag: bool = Var(init=True)

        def update(self) -> State:
            return self.State(flag=True)

    mode = Mode()
    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "start",
                nodes=(mode,),
                transitions=(
                    If(V(Mode.State.flag), terminate, name="done"),
                    If(~V(Mode.State.flag), terminate, name="skip"),
                ),
                is_initial=True,
            )
        ],
    )

    system.step()

    assert system.history[0].phase == "start"


def test_guard_expression_accepts_state_reference_compared_to_float() -> None:
    class Level(Node):
        class State(NodeState):
            value: float = Var(init=0.0)

        def update(self) -> State:
            return self.State(value=1.25)

    class High(Node):
        class State(NodeState):
            reached: bool = Var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    class Low(Node):
        class State(NodeState):
            reached: bool = Var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    level = Level()
    high = High()
    low = Low()
    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "measure",
                nodes=(level,),
                transitions=(
                    If(V(Level.State.value) > 1.0, "high", name="high"),
                    If(V(Level.State.value) <= 1.0, "low", name="low"),
                ),
                is_initial=True,
            ),
            Phase("high", nodes=(high,), transitions=(Goto(terminate),)),
            Phase("low", nodes=(low,), transitions=(Goto(terminate),)),
        ],
    )

    system.step()

    assert [record.phase for record in system.history] == ["measure", "high"]
    assert system.snapshot()["High.reached"] is True
    assert system.snapshot()["Low.reached"] is False


def test_enum_state_vars_are_supported_in_symbolic_guards() -> None:
    class ModeValue(Enum):
        IDLE = "idle"
        ACTIVE = "active"
        FAILED = "failed"

    class Mode(Node):
        class State(NodeState):
            value: ModeValue = Var(init=ModeValue.IDLE)

        def update(self) -> State:
            return self.State(value=ModeValue.ACTIVE)

    class Active(Node):
        class State(NodeState):
            reached: bool = Var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    class Inactive(Node):
        class State(NodeState):
            reached: bool = Var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    mode = Mode()
    active = Active()
    inactive = Inactive()
    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "mode",
                nodes=(mode,),
                transitions=(
                    If(V(Mode.State.value) == ModeValue.ACTIVE, "active", name="active"),
                    If(V(Mode.State.value) != ModeValue.ACTIVE, "inactive", name="inactive"),
                ),
                is_initial=True,
            ),
            Phase("active", nodes=(active,), transitions=(Goto(terminate),)),
            Phase("inactive", nodes=(inactive,), transitions=(Goto(terminate),)),
        ],
    )

    assert system.compile_report.ok
    assert system.snapshot()["Mode.value"] is ModeValue.IDLE

    system.step()

    assert [record.phase for record in system.history] == ["mode", "active"]
    assert system.snapshot()["Mode.value"] is ModeValue.ACTIVE
    assert system.snapshot()["Active.reached"] is True
    assert system.snapshot()["Inactive.reached"] is False


def test_transition_target_accepts_phase_instance() -> None:
    class Source(Node):
        class State(NodeState):
            ready: bool = Var(init=True)

        def update(self) -> State:
            return self.State(ready=True)

    class Done(Node):
        class State(NodeState):
            reached: bool = Var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    source = Source()
    done_node = Done()
    done = Phase("done", nodes=(done_node,), transitions=(Goto(terminate),))
    start = Phase(
        "start",
        nodes=(source,),
        transitions=(
            If(V(Source.State.ready), done, name="ready"),
            If(~V(Source.State.ready), terminate, name="not-ready"),
        ),
        is_initial=True,
    )
    system = PhasedReactiveSystem(phases=[start, done])

    system.step()

    assert [record.phase for record in system.history] == ["start", "done"]
    assert system.snapshot()["Done.reached"] is True


def test_phase_transition_guards_may_read_state_vars_outside_active_phase() -> None:
    class Source(Node):
        class State(NodeState):
            ready: bool = Var(init=True)

        def update(self) -> State:
            return self.State(ready=True)

    class Other(Node):
        class State(NodeState):
            value: int = Var(init=0)

        def update(self) -> State:
            return self.State(value=1)

    source = Source()
    other = Other()
    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "check",
                nodes=(other, source),
                transitions=(
                    If(V(Source.State.ready), terminate, name="ready"),
                    If(~V(Source.State.ready), terminate, name="not-ready"),
                ),
                is_initial=True,
            )
        ],
    )

    assert system.compile_report.ok
    system.step()
    assert system.snapshot()["Other.value"] == 1


def test_phase_transition_guard_accepts_instance_state_reference() -> None:
    class Source(Node):
        class State(NodeState):
            ready: bool = Var(init=True)

    source = Source(name="source_a")
    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "check",
                nodes=(source,),
                transitions=(
                    If(V(source.State.ready), terminate, name="ready"),
                    If(~V(source.State.ready), terminate, name="not-ready"),
                ),
                is_initial=True,
            )
        ],
    )

    assert system.compile_report.ok


def test_phase_transition_guard_rejects_ambiguous_class_state_reference() -> None:
    class Source(Node):
        class State(NodeState):
            ready: bool = Var(init=True)

    source_a = Source(name="source_a")
    source_b = Source(name="source_b")

    with pytest.raises(CompileError) as exc_info:
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "check",
                    nodes=(source_a, source_b),
                    transitions=(
                        If(V(Source.State.ready), terminate, name="ready"),
                        If(~V(Source.State.ready), terminate, name="not-ready"),
                    ),
                    is_initial=True,
                )
            ],
        )

    assert any(
        issue.location == "check.ready"
        and "ambiguous guard variable 'Source.ready'" in issue.message
        and "source_a.ready" in issue.message
        and "source_b.ready" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_connect_supports_instance_node_identity() -> None:
    class Source(Node):
        class State(NodeState):
            value: int = Var(init=0)

        def __init__(self, value: int, *, name: str | None = None) -> None:
            super().__init__(name=name)
            self.value = value

        def update(self) -> State:
            return self.State(value=self.value)

    class Sink(Node):
        class Inputs(NodeInputs):
            value: int = Input()

        class State(NodeState):
            seen: int = Var(init=0)

        def update(self, inputs: Inputs) -> State:
            return self.State(seen=inputs.value)

    source_a = Source(1, name="source_a")
    source_b = Source(2, name="source_b")
    sink_a = Sink(name="sink_a")
    sink_b = Sink(name="sink_b")

    port(sink_a.Inputs.value).connect(source_a.State.value)
    port(source_b.State.value).connect(sink_b.Inputs.value)

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "copy",
                nodes=(source_a, source_b, sink_a, sink_b),
                transitions=(Goto(terminate),),
                is_initial=True,
            )
        ],
    )

    assert system.compile_report.nodes == (
        "source_a",
        "source_b",
        "sink_a",
        "sink_b",
    )
    assert system.compile_report.inputs["sink_a.value"] == "source_a.value"
    assert system.compile_report.inputs["sink_b.value"] == "source_b.value"

    system.step()
    snapshot = system.snapshot()

    assert snapshot["sink_a.seen"] == 1
    assert snapshot["sink_b.seen"] == 2


def test_connect_rejects_non_input_left_side() -> None:
    class Source(Node):
        class State(NodeState):
            value: int = Var(init=0)

    source = Source()

    with pytest.raises(TypeError, match="input port on the left side"):
        port(source.State.value).connect(source.State.value)


def test_connect_rejects_non_state_right_side() -> None:
    class Sink(Node):
        class Inputs(NodeInputs):
            value: int = Input()

    sink = Sink()

    with pytest.raises(TypeError, match="state port, state reference"):
        port(sink.Inputs.value).connect(sink.Inputs.value)


def test_node_name_can_default_from_class_or_instance_override() -> None:
    class NamedSource(Node):
        name = "class_named_source"

        class State(NodeState):
            value: int = Var(init=1)

    class PlainSource(Node):
        class State(NodeState):
            value: int = Var(init=2)

    system = _tick_system(
        [
            NamedSource(),
            PlainSource(name="instance_named_source"),
        ]
    )

    assert system.compile_report.nodes == (
        "class_named_source",
        "instance_named_source",
    )
    assert system.snapshot() == {
        "class_named_source.value": 1,
        "instance_named_source.value": 2,
    }


def test_implicit_node_names_are_deduplicated() -> None:
    class Source(Node):
        class State(NodeState):
            value: int = Var(init=1)

    system = _tick_system([Source(), Source()])

    assert system.compile_report.nodes == ("Source", "Source_2")
    assert system.snapshot() == {
        "Source.value": 1,
        "Source_2.value": 1,
    }


def test_explicit_duplicate_node_names_are_rejected() -> None:
    class Source(Node):
        class State(NodeState):
            value: int = Var(init=1)

    with pytest.raises(CompileError) as exc_info:
        _tick_system([Source(name="source"), Source(name="source")])

    assert any(
        issue.location == "source" and issue.message == "node name is declared more than once"
        for issue in exc_info.value.report.issues
    )


def test_class_level_source_is_rejected_when_multiple_instances_exist() -> None:
    class Plant(Node):
        class State(NodeState):
            theta: float = Var(init=0.0)

    class Controller(Node):
        class Inputs(NodeInputs):
            theta: float = Input(src=Plant.State.theta)

        class State(NodeState):
            torque: float = Var(init=0.0)

    with pytest.raises(CompileError) as exc_info:
        _tick_system([Plant(), Plant(), Controller()])

    assert any(
        issue.location == "Controller.theta"
        and "ambiguous input source 'Plant.theta'" in issue.message
        and "Plant_2.theta" in issue.message
        and "use instance connection" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_phase_predicate_routing() -> None:
    class Source(Node):
        class State(NodeState):
            value: float = Var(init=1.0)

        def update(self, inputs: NodeInputs) -> State:
            return self.State(value=2.0)

    class Sink(Node):
        class Inputs(NodeInputs):
            value: float = Input(src=Source.State.value)

        class State(NodeState):
            seen: float = Var(init=0.0)

        def update(self, inputs: Inputs) -> State:
            return self.State(seen=inputs.value)

    source = Source()
    sink = Sink()
    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "source",
                nodes=(source,),
                transitions=(
                    If(
                        lambda state: state["Source.value"] > 1.0,
                        "sink",
                        name="value-ready",
                    ),
                ),
                is_initial=True,
            ),
            Phase("sink", nodes=(sink,), transitions=(Goto(terminate),)),
        ],
    )

    system.step()

    assert system.snapshot()["Sink.seen"] == 2.0


def test_compile_rejects_c3_ambiguous_predicates() -> None:
    with pytest.raises(CompileError) as exc_info:
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "start",
                    nodes=(),
                    transitions=(
                        If(lambda _: True, terminate, name="first"),
                        If(lambda _: True, terminate, name="second"),
                    ),
                    is_initial=True,
                )
            ],
        )

    assert any(
        issue.location == "start" and issue.message.startswith("C3 violation")
        for issue in exc_info.value.report.issues
    )


def test_compile_rejects_c3_example() -> None:
    with pytest.raises(CompileError) as exc_info:
        build_c3_violation_system()

    assert any(
        issue.location == "ambiguous" and issue.message.startswith("C3 violation")
        for issue in exc_info.value.report.issues
    )


def test_if_elseif_else_chain_uses_ordered_fallback_semantics() -> None:
    class Mode(Node):
        class State(NodeState):
            value: int = Var(init=2, domain=(0, 1, 2))

        def update(self) -> State:
            return self.State(value=2)

    class One(Node):
        class State(NodeState):
            reached: bool = Var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    class Two(Node):
        class State(NodeState):
            reached: bool = Var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    class Fallback(Node):
        class State(NodeState):
            reached: bool = Var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    mode = Mode()
    one = One()
    two = Two()
    fallback = Fallback()
    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "select",
                nodes=(mode,),
                transitions=(
                    If(V(Mode.State.value) == 1, "one", name="one"),
                    ElseIf(V(Mode.State.value) == 2, "two", name="two"),
                    Else("fallback"),
                ),
                is_initial=True,
            ),
            Phase("one", nodes=(one,), transitions=(Goto(terminate),)),
            Phase("two", nodes=(two,), transitions=(Goto(terminate),)),
            Phase("fallback", nodes=(fallback,), transitions=(Goto(terminate),)),
        ],
    )

    system.step()

    assert system.snapshot()["Two.reached"] is True
    assert system.snapshot()["One.reached"] is False
    assert system.snapshot()["Fallback.reached"] is False


def test_if_chain_order_gives_precedence_over_later_elseif() -> None:
    class Mode(Node):
        class State(NodeState):
            value: int = Var(init=2, domain=(0, 1, 2))

        def update(self) -> State:
            return self.State(value=2)

    class First(Node):
        class State(NodeState):
            reached: bool = Var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    class Second(Node):
        class State(NodeState):
            reached: bool = Var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    mode = Mode()
    first = First()
    second = Second()
    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "select",
                nodes=(mode,),
                transitions=(
                    If(V(Mode.State.value) > 0, "first", name="first"),
                    ElseIf(V(Mode.State.value) > 1, "second", name="second"),
                    Else(terminate),
                ),
                is_initial=True,
            ),
            Phase("first", nodes=(first,), transitions=(Goto(terminate),)),
            Phase("second", nodes=(second,), transitions=(Goto(terminate),)),
        ],
    )

    system.step()

    assert system.snapshot()["First.reached"] is True
    assert system.snapshot()["Second.reached"] is False


def test_multiple_if_chains_overlap_is_c3_violation() -> None:
    class Mode(Node):
        class State(NodeState):
            a: bool = Var(init=True)
            b: bool = Var(init=True)

    with pytest.raises(CompileError) as exc_info:
        mode = Mode()
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "select",
                    nodes=(mode,),
                    transitions=(
                        If(V(Mode.State.a), terminate, name="a"),
                        If(V(Mode.State.b), terminate, name="b"),
                    ),
                    is_initial=True,
                )
            ],
        )

    assert any(
        issue.location == "select" and issue.message.startswith("C3 violation")
        for issue in exc_info.value.report.issues
    )


def test_multiple_if_chains_can_be_disjoint() -> None:
    class Mode(Node):
        class State(NodeState):
            a: bool = Var(init=False)

        def update(self) -> State:
            return self.State(a=False)

    mode = Mode()
    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "select",
                nodes=(mode,),
                transitions=(
                    If(V(Mode.State.a), terminate, name="a"),
                    If(~V(Mode.State.a), terminate, name="not-a"),
                ),
                is_initial=True,
            )
        ],
    )

    assert system.compile_report.ok
    system.step()


def test_goto_is_unconditional() -> None:
    class Source(Node):
        class State(NodeState):
            value: int = Var(init=0)

        def update(self) -> State:
            return self.State(value=1)

    class Sink(Node):
        class State(NodeState):
            reached: bool = Var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    source = Source()
    sink = Sink()
    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "source",
                nodes=(source,),
                transitions=(Goto("sink"),),
                is_initial=True,
            ),
            Phase("sink", nodes=(sink,), transitions=(Goto(terminate),)),
        ],
    )

    system.step()

    assert system.snapshot()["Source.value"] == 1
    assert system.snapshot()["Sink.reached"] is True


def test_goto_cannot_be_mixed_with_if_chains() -> None:
    class Source(Node):
        class State(NodeState):
            flag: bool = Var(init=True)

    with pytest.raises(CompileError) as exc_info:
        source = Source()
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "mixed",
                    nodes=(source,),
                    transitions=(
                        If(V(Source.State.flag), terminate),
                        Goto(terminate),
                    ),
                    is_initial=True,
                )
            ],
        )

    assert any(
        issue.message == "Goto transitions cannot be mixed with If/ElseIf/Else chains"
        for issue in exc_info.value.report.issues
    )


def test_elseif_and_else_require_open_if_chain() -> None:
    class Source(Node):
        class State(NodeState):
            flag: bool = Var(init=True)

    with pytest.raises(CompileError) as exc_info:
        source = Source()
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "bad",
                    nodes=(source,),
                    transitions=(
                        ElseIf(V(Source.State.flag), terminate),
                        Else(terminate),
                    ),
                    is_initial=True,
                )
            ],
        )

    messages = [issue.message for issue in exc_info.value.report.issues]
    assert "ElseIf must follow If or ElseIf" in messages
    assert "Else must follow If or ElseIf" in messages


def test_elseif_after_else_is_compile_error() -> None:
    class Source(Node):
        class State(NodeState):
            flag: bool = Var(init=True)

    with pytest.raises(CompileError) as exc_info:
        source = Source()
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "bad",
                    nodes=(source,),
                    transitions=(
                        If(V(Source.State.flag), terminate),
                        Else(terminate),
                        ElseIf(~V(Source.State.flag), terminate),
                    ),
                    is_initial=True,
                )
            ],
        )

    assert any(
        issue.location == "bad.elseif" and issue.message == "ElseIf must follow If or ElseIf"
        for issue in exc_info.value.report.issues
    )


def test_second_else_is_compile_error() -> None:
    class Source(Node):
        class State(NodeState):
            flag: bool = Var(init=True)

    with pytest.raises(CompileError) as exc_info:
        source = Source()
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "bad",
                    nodes=(source,),
                    transitions=(
                        If(V(Source.State.flag), terminate),
                        Else(terminate, name="first-else"),
                        Else(terminate, name="second-else"),
                    ),
                    is_initial=True,
                )
            ],
        )

    assert any(
        issue.location == "bad.second-else" and issue.message == "Else must follow If or ElseIf"
        for issue in exc_info.value.report.issues
    )


def test_if_chain_without_total_coverage_is_c3_violation() -> None:
    class Source(Node):
        class State(NodeState):
            flag: bool = Var(init=False)

    with pytest.raises(CompileError) as exc_info:
        source = Source()
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "select",
                    nodes=(source,),
                    transitions=(If(V(Source.State.flag), terminate),),
                    is_initial=True,
                )
            ],
        )

    assert any(
        issue.location == "select"
        and issue.message.startswith("C3 violation: no transition is enabled")
        for issue in exc_info.value.report.issues
    )


def test_if_after_else_is_reported_as_warning() -> None:
    class Mode(Node):
        class State(NodeState):
            a: bool = Var(init=False)
            b: bool = Var(init=True)

    mode = Mode()
    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "select",
                nodes=(mode,),
                transitions=(
                    If(V(Mode.State.a), terminate, name="a"),
                    Else("next", name="else-a"),
                    If(V(Mode.State.b), terminate, name="b"),
                ),
                is_initial=True,
            ),
            Phase("next", nodes=(), transitions=(Goto(terminate),)),
        ],
        strict=False,
    )

    assert system.compile_report.ok
    assert not system.compile_report.issues
    assert any(
        issue.location == "select.b" and issue.message.startswith("transition follows Else")
        for issue in system.compile_report.warnings
    )


def test_if_after_else_starts_new_chain_that_can_have_else() -> None:
    class Mode(Node):
        class State(NodeState):
            a: bool = Var(init=False)
            b: bool = Var(init=True)

    mode = Mode()
    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "select",
                nodes=(mode,),
                transitions=(
                    If(V(Mode.State.a), terminate, name="a"),
                    Else("next", name="else-a"),
                    If(V(Mode.State.b), terminate, name="b"),
                    Else(terminate, name="else-b"),
                ),
                is_initial=True,
            ),
            Phase("next", nodes=(), transitions=(Goto(terminate),)),
        ],
        strict=False,
    )

    assert system.compile_report.ok
    assert any(
        issue.location == "select.b" and issue.message.startswith("transition follows Else")
        for issue in system.compile_report.warnings
    )


def test_compile_rejects_multiple_initial_phases() -> None:
    with pytest.raises(CompileError) as exc_info:
        PhasedReactiveSystem(
            phases=[
                Phase("one", nodes=(), transitions=(Goto(terminate),), is_initial=True),
                Phase("two", nodes=(), transitions=(Goto(terminate),), is_initial=True),
            ],
        )

    assert not exc_info.value.report.ok
    assert any(
        issue.message == "exactly one phase must be marked initial"
        for issue in exc_info.value.report.issues
    )


def test_compile_rejects_missing_initial_phase_marker() -> None:
    with pytest.raises(CompileError) as exc_info:
        PhasedReactiveSystem(
            phases=[
                Phase("one", nodes=(), transitions=(Goto(terminate),)),
                Phase("two", nodes=(), transitions=(Goto(terminate),)),
            ],
        )

    assert not exc_info.value.report.ok
    assert any(
        issue.message == "exactly one phase must be marked initial"
        for issue in exc_info.value.report.issues
    )


def test_compile_rejects_unreachable_phase() -> None:
    with pytest.raises(CompileError) as exc_info:
        PhasedReactiveSystem(
            phases=[
                Phase("start", nodes=(), transitions=(Goto(terminate),), is_initial=True),
                Phase("orphan", nodes=(), transitions=(Goto(terminate),)),
            ],
        )

    assert any(
        issue.location == "orphan"
        and issue.message == "phase is unreachable from initial phase 'start'"
        for issue in exc_info.value.report.issues
    )


def test_compile_report_records_unreachable_phase_when_not_strict() -> None:
    system = PhasedReactiveSystem(
        phases=[
            Phase("start", nodes=(), transitions=(Goto(terminate),), is_initial=True),
            Phase("orphan", nodes=(), transitions=(Goto(terminate),)),
        ],
        strict=False,
    )

    assert not system.compile_report.ok
    assert any(
        issue.location == "orphan"
        and issue.message == "phase is unreachable from initial phase 'start'"
        for issue in system.compile_report.issues
    )


def test_compile_rejects_c1_phase_cycle() -> None:
    with pytest.raises(CompileError) as exc_info:
        build_c1_violation_system()

    assert any(
        issue.location == "coupled" and issue.message.startswith("C1 violation")
        for issue in exc_info.value.report.issues
    )


def test_complex_ok_system_compiles_and_runs() -> None:
    system = build_complex_ok_system()

    assert system.compile_report.ok
    system.run(steps=3)

    assert len(system.snapshot()["TraceLogger.trace"]) == 3


def test_complex_bad_system_fails_c1() -> None:
    with pytest.raises(CompileError) as exc_info:
        build_bad_c1_system()

    assert any(
        issue.location == "bad-coupled-control" and issue.message.startswith("C1 violation")
        for issue in exc_info.value.report.issues
    )


def test_complex_c3_ok_system_compiles() -> None:
    system = build_c3_ok_system()

    assert system.compile_report.ok


def test_complex_c3_bad_system_fails_partition() -> None:
    with pytest.raises(CompileError) as exc_info:
        build_bad_overlap_system()

    assert any(
        issue.location == "diagnose" and issue.message.startswith("C3 violation")
        for issue in exc_info.value.report.issues
    )


def test_c2_cycle_with_local_guard_node_is_rejected_when_c2star_is_feasible() -> None:
    class Mode(Node):
        class State(NodeState):
            flag: bool = Var(init=False)

    with pytest.raises(CompileError) as exc_info:
        mode = Mode()
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "a",
                    nodes=(mode,),
                    transitions=(
                        If(V("Mode.flag"), "b", name="to-b"),
                        If(~V("Mode.flag"), terminate, name="stop-a"),
                    ),
                    is_initial=True,
                ),
                Phase(
                    "b",
                    nodes=(mode,),
                    transitions=(
                        If(~V("Mode.flag"), "a", name="to-a"),
                        If(V("Mode.flag"), terminate, name="stop-b"),
                    ),
                ),
            ],
        )

    assert any("C2*" in issue.message for issue in exc_info.value.report.issues)


def test_c2star_rejects_feasible_cycle() -> None:
    class Mode(Node):
        class State(NodeState):
            flag: bool = Var(init=False)

    with pytest.raises(CompileError) as exc_info:
        mode = Mode()
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "a",
                    nodes=(mode,),
                    transitions=(
                        If(V("Mode.flag"), "b", name="to-b"),
                        If(~V("Mode.flag"), terminate, name="stop-a"),
                    ),
                    is_initial=True,
                ),
                Phase(
                    "b",
                    nodes=(mode,),
                    transitions=(
                        If(V("Mode.flag"), "a", name="to-a"),
                        If(~V("Mode.flag"), terminate, name="stop-b"),
                    ),
                ),
            ],
        )

    assert any("C2*" in issue.message for issue in exc_info.value.report.issues)


def test_c2star_uses_effective_elseif_guards() -> None:
    class Mode(Node):
        class State(NodeState):
            flag: bool = Var(init=False)
            gate: bool = Var(init=False)

    with pytest.raises(CompileError) as exc_info:
        mode = Mode()
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "a",
                    nodes=(mode,),
                    transitions=(
                        If(V("Mode.flag"), terminate, name="done-a"),
                        ElseIf(V("Mode.gate"), "b", name="to-b"),
                        Else(terminate, name="stop-a"),
                    ),
                    is_initial=True,
                ),
                Phase(
                    "b",
                    nodes=(mode,),
                    transitions=(
                        If(V("Mode.flag"), terminate, name="done-b"),
                        ElseIf(V("Mode.gate"), "a", name="to-a"),
                        Else(terminate, name="stop-b"),
                    ),
                ),
            ],
        )

    assert any(
        "C2*" in issue.message and "a -> b -> a" in issue.location
        for issue in exc_info.value.report.issues
    )


def test_mohanty2023_fig4_conflict_resolution() -> None:
    system = build_fig4_system()

    assert system.compile_report.ok
    system.step()

    snapshot = system.snapshot()
    assert snapshot["FireDetectionSystem.fds_state"] == S2
    assert snapshot["LeakDetectionSystem.lds_state"] == LS1
    assert snapshot["ConflictResolver.system_substate"] == SS0
    assert snapshot["ActionDispatcher.sprinkler_on"] is True
    assert snapshot["ActionDispatcher.water_valve_closed"] is False


def test_safe_recovery_pipeline_cycle_is_rejected_under_local_guard_rule() -> None:
    with pytest.raises(CompileError) as exc_info:
        build_safe_recovery_system()

    assert any("C2*" in issue.message for issue in exc_info.value.report.issues)


def test_c2star_split_writes_compile_without_phase_guard_locality_rule() -> None:
    system = build_c2star_split_writes_system(p_x=0.0, p_y=1.0, seed=0)

    assert system.compile_report.ok


def test_smart_home_v1_fails_c1() -> None:
    with pytest.raises(CompileError) as exc_info:
        build_smart_home_v1()
    assert any(
        "C1 violation" in issue.message and "classify" in issue.location
        for issue in exc_info.value.report.issues
    )


def test_smart_home_v2_fails_c2star() -> None:
    with pytest.raises(CompileError) as exc_info:
        build_smart_home_v2()
    assert any(
        "C2*" in issue.message and "command -> verify" in issue.location
        for issue in exc_info.value.report.issues
    )


def test_smart_home_v3_counter_does_not_save_c2star() -> None:
    with pytest.raises(CompileError) as exc_info:
        build_smart_home_v3()
    assert any(
        "C2*" in issue.message and "AttemptCounter.attempts" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_smart_home_v4_compiles_and_resolves_fire_over_leak() -> None:
    system = build_smart_home_v4(max_attempts=3)
    assert system.compile_report.ok

    system.step()
    snapshot = system.snapshot()
    assert snapshot["FireClassifier.fire_state"] == 2
    assert snapshot["LeakClassifier.leak_state"] == 1
    assert snapshot["PriorityArbiter.action_plan"] == ACT_FIRE_OVER_LEAK
    assert snapshot["SprinklerCmd.sprinkler_on"] is True
    assert snapshot["ValveCmd.valve_closed"] is True
    assert snapshot["AlarmCmd.alarm_on"] is True
    assert snapshot["NotifCmd.notif_sent"] is True


def test_smart_home_v4_max_attempts_controls_phase_count() -> None:
    small = build_smart_home_v4(max_attempts=1)
    assert small.compile_report.ok
    assert len(small.phases) == 4  # sense, classify, command_1, verify_1

    big = build_smart_home_v4(max_attempts=5)
    assert big.compile_report.ok
    assert len(big.phases) == 12  # sense, classify, 5 * (command, verify)


def test_async_world_compiles_with_oracles() -> None:
    system = build_async_world(max_attempts=3)
    assert system.compile_report.ok
    # env, sense, staging, classify, 3 * (command, verify)
    assert len(system.phases) == 10

    system.step()
    snapshot = system.snapshot()
    # deterministic happy-path: all oracles up, smoke+heat+leak all positive
    assert snapshot["AsyncFireClassifier.fire_state"] == 2
    assert snapshot["AsyncLeakClassifier.leak_state"] == 1
    assert snapshot["AsyncPriorityArbiter.action_plan"] == ACT_FIRE_OVER_LEAK
    assert snapshot["SprinklerCmdAsync.sprinkler_on"] is True
    assert snapshot["ValveCmdAsync.valve_closed"] is True


def test_async_world_with_cycle_rejected_by_c2star() -> None:
    with pytest.raises(CompileError) as exc_info:
        build_async_world_with_cycle()
    assert any(
        "C2*" in issue.message and "command -> verify" in issue.location
        for issue in exc_info.value.report.issues
    )
