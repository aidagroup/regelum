import pytest

import regelum as rg
from tests.core.system._support import _tick_system


def test_compile_rejects_unknown_input_source() -> None:
    class Broken(rg.Node):
        class Inputs(rg.NodeInputs):
            value: float = rg.src("Missing.State.value")

        class State(rg.NodeState):
            result: float = rg.var(init=0.0)

    with pytest.raises(rg.CompileError) as exc_info:
        _tick_system([Broken()])

    assert not exc_info.value.report.ok
    assert exc_info.value.report.issues[0].location == "Broken.value"


def test_compile_rejects_unconnected_input() -> None:
    class Broken(rg.Node):
        class Inputs(rg.NodeInputs):
            value: float = rg.src()

        class State(rg.NodeState):
            result: float = rg.var(init=0.0)

    with pytest.raises(rg.CompileError) as exc_info:
        _tick_system([Broken()])

    assert not exc_info.value.report.ok
    assert exc_info.value.report.issues[0].location == "Broken.value"


def test_compile_rejects_duplicate_state_paths() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            value: float = rg.var(init=0.0)

    with pytest.raises(rg.CompileError) as exc_info:
        _tick_system([Source(name="source"), Source(name="source")])

    assert not exc_info.value.report.ok
    assert exc_info.value.report.issues[0].location == "source.value"


def test_compile_rejects_missing_state_initial_value_before_first_read() -> None:
    class Source(rg.Node):
        class Inputs(rg.NodeInputs):
            previous: float = rg.src(lambda: Source.State.value)

        class State(rg.NodeState):
            value: float = rg.var()

        def update(self, inputs: Inputs) -> State:
            return self.State(value=inputs.previous + 1)

    with pytest.raises(rg.CompileError) as exc_info:
        _tick_system([Source()])

    assert not exc_info.value.report.ok
    assert exc_info.value.report.issues[0].location == "Source.value"
    assert exc_info.value.report.state_vars_without_initial == ("Source.value",)
    assert exc_info.value.report.required_initial_state_vars == {
        "Source.value": ("Source.previous",)
    }
    assert (
        exc_info.value.report.issues[0].message
        == "state variable initial value is required before first read by ('Source.previous',)"
    )


def test_state_var_without_initial_is_allowed_when_written_before_read() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var()

        def update(self) -> State:
            return self.State(value=5)

    class Sink(rg.Node):
        class Inputs(rg.NodeInputs):
            value: int = rg.src(lambda: Source.State.value)

        class State(rg.NodeState):
            seen: int = rg.var(init=0)

        def update(self, inputs: Inputs) -> State:
            return self.State(seen=inputs.value)

    source = Source()
    sink = Sink()
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "copy",
                nodes=(sink, source),
                transitions=(rg.Goto(rg.terminate),),
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
    class Broken(rg.Node):
        class Inputs(rg.NodeInputs):
            value: int

        class State(rg.NodeState):
            seen: int

        def update(self, inputs: Inputs) -> State:
            return self.State(seen=inputs.value)

    system = _tick_system([Broken()], strict=False)

    assert not system.compile_report.ok
    assert system.compile_report.unlinked_inputs == ("Broken.value",)
    assert system.compile_report.minimal_initial_state_vars == ()
    assert system.compile_report.state_vars_without_initial == ("Broken.seen",)
