import pytest

import regelum as rg
from tests.core.node_resolution._support import Flag


def test_multi_phase_pipeline_resolves_instance_inputs_and_guards_across_branches() -> None:
    class Sensor(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=0)
            ready: bool = rg.var(init=True)

        def __init__(self, value: int, *, name: str) -> None:
            super().__init__(name=name)
            self.value = value

        def update(self) -> State:
            return self.State(value=self.value, ready=True)

    class Controller(rg.Node):
        class Inputs(rg.NodeInputs):
            value: int = rg.src()

        class State(rg.NodeState):
            command: int = rg.var()
            high: bool = rg.var(init=True)

        def update(self, inputs: Inputs) -> State:
            return self.State(command=inputs.value + 10, high=inputs.value > 5)

    class Actuator(rg.Node):
        class Inputs(rg.NodeInputs):
            command: int = rg.src()

        class State(rg.NodeState):
            applied: int = rg.var(init=0)

        def update(self, inputs: Inputs) -> State:
            return self.State(applied=inputs.command)

    primary = Sensor(7, name="primary")
    backup = Sensor(2, name="backup")
    controller = Controller(name="controller")
    actuator = Actuator(name="actuator")
    rg.port(controller.Inputs.value).connect(primary.State.value)
    rg.port(actuator.Inputs.command).connect(controller.State.command)

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "sense",
                nodes=(primary, backup),
                transitions=(
                    rg.If(rg.V(primary.State.ready), "control", name="primary-ready"),
                    rg.ElseIf(rg.V(backup.State.ready), "control", name="backup-ready"),
                    rg.Else(rg.terminate, name="no-sensor"),
                ),
                is_initial=True,
            ),
            rg.Phase(
                "control",
                nodes=(controller,),
                transitions=(
                    rg.If(rg.V(controller.State.high), "actuate", name="high"),
                    rg.Else(rg.terminate, name="low"),
                ),
            ),
            rg.Phase("actuate", nodes=(actuator,), transitions=(rg.Goto(rg.terminate),)),
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
    class ExternalConfig(rg.Node):
        class State(rg.NodeState):
            enabled: bool = rg.var(init=True)
            gain: int = rg.var(init=2)

    class ExternalMode(rg.Node):
        class State(rg.NodeState):
            armed: bool = rg.var(init=True)

    class Processor(rg.Node):
        class Inputs(rg.NodeInputs):
            gain: int = rg.src()

        class State(rg.NodeState):
            score: int = rg.var(init=0)

        def update(self, inputs: Inputs) -> State:
            return self.State(score=inputs.gain)

    config = ExternalConfig(name="config")
    mode = ExternalMode(name="mode")
    processor = Processor(name="processor")
    rg.port(processor.Inputs.gain).connect(config.State.gain)

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "process",
                nodes=(processor,),
                transitions=(
                    rg.If(
                        rg.V(config.State.enabled) & rg.V(mode.State.armed),
                        rg.terminate,
                        name="external-ready",
                    ),
                    rg.Else(rg.terminate, name="fallback"),
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
    class Mode(rg.Node):
        class State(rg.NodeState):
            ready: bool = rg.var(init=True)

    class Work(rg.Node):
        class State(rg.NodeState):
            done: bool = rg.var(init=False)

    with pytest.raises(rg.CompileError) as exc_info:
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "left",
                    nodes=(Mode(name="left_mode"),),
                    transitions=(rg.Goto("right"),),
                    is_initial=True,
                ),
                rg.Phase(
                    "right",
                    nodes=(Mode(name="right_mode"),),
                    transitions=(rg.Goto("join"),),
                ),
                rg.Phase(
                    "join",
                    nodes=(Work(),),
                    transitions=(
                        rg.If(rg.V(Mode.State.ready), rg.terminate, name="ready"),
                        rg.Else(rg.terminate, name="not-ready"),
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
    class Mode(rg.Node):
        class State(rg.NodeState):
            ready: bool = rg.var(init=True)

        def update(self) -> State:
            return self.State(ready=True)

    class Work(rg.Node):
        class State(rg.NodeState):
            done: bool = rg.var(init=False)

        def update(self) -> State:
            return self.State(done=True)

    left_mode = Mode(name="left_mode")
    right_mode = Mode(name="right_mode")
    work = Work()

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "left",
                nodes=(left_mode,),
                transitions=(rg.Goto("right"),),
                is_initial=True,
            ),
            rg.Phase("right", nodes=(right_mode,), transitions=(rg.Goto("join"),)),
            rg.Phase(
                "join",
                nodes=(work,),
                transitions=(
                    rg.If(rg.V(right_mode.State.ready), rg.terminate, name="ready"),
                    rg.Else(rg.terminate, name="not-ready"),
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

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "route",
                nodes=(gate_a, gate_b, gate_c),
                transitions=(
                    rg.If(rg.V(gate_a.State.ready), "a", name="a"),
                    rg.ElseIf(rg.V(gate_b.State.ready), "b", name="b"),
                    rg.Else(rg.terminate, name="fallback"),
                    rg.If(rg.V(missing_gate.State.ready), "missing", name="missing"),
                    rg.ElseIf(rg.V(gate_c.State.ready), "c", name="c"),
                    rg.Else(rg.terminate, name="second-fallback"),
                ),
                is_initial=True,
            ),
            rg.Phase("a", nodes=(), transitions=(rg.Goto(rg.terminate),)),
            rg.Phase("b", nodes=(), transitions=(rg.Goto(rg.terminate),)),
            rg.Phase("c", nodes=(), transitions=(rg.Goto(rg.terminate),)),
            rg.Phase("missing", nodes=(), transitions=(rg.Goto(rg.terminate),)),
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
