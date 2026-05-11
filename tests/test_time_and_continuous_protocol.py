from __future__ import annotations

from fractions import Fraction
from typing import Any, cast

import pytest

from regelum import (
    Clock,
    CompileError,
    Else,
    Goto,
    If,
    Input,
    Node,
    NodeInputs,
    NodeState,
    ODENode,
    ODESystem,
    Phase,
    PhasedReactiveSystem,
    V,
    Var,
    terminate,
)


def _one_phase_system(
    *nodes: Node,
    base_dt: Fraction | int | str = "auto",
) -> PhasedReactiveSystem:
    return PhasedReactiveSystem(
        phases=[
            Phase(
                "tick",
                nodes=nodes,
                transitions=(Goto(terminate),),
                is_initial=True,
            )
        ],
        base_dt=base_dt,
    )


class Counter(Node):
    class Inputs(NodeInputs):
        previous: int = Input(src=lambda: Counter.State.value)

    class State(NodeState):
        value: int = Var(init=0)

    def update(self, inputs: Inputs) -> State:
        return self.State(value=inputs.previous + 1)


class ScheduledCounter(Node):
    def __init__(self, *, name: str | None = None, dt: Fraction | int | str | None = None) -> None:
        super().__init__(name=name, dt=dt)
        self.count = 0

    class State(NodeState):
        value: int = Var(init=0)

    def update(self) -> State:
        self.count += 1
        return self.State(value=self.count)


class ClockLog(Node):
    class Inputs(NodeInputs):
        previous_ticks: tuple[int, ...] = Input(src=lambda: ClockLog.State.ticks)
        previous_times: tuple[float, ...] = Input(src=lambda: ClockLog.State.times)
        tick: int = Input(src=Clock.tick)
        time: float = Input(src=Clock.time)

    class State(NodeState):
        ticks: tuple[int, ...] = Var(init=())
        times: tuple[float, ...] = Var(init=())

    def update(self, inputs: Inputs) -> State:
        return self.State(
            ticks=(*inputs.previous_ticks, inputs.tick),
            times=(*inputs.previous_times, inputs.time),
        )


class TimeDrivenIntegrator(ODENode):
    class State(NodeState):
        x: float = Var(init=0.0)

    def dstate(  # ty: ignore[invalid-method-override]
        self,
        inputs: NodeInputs,
        state: State,
        *,
        time: Any,
    ) -> State:
        return self.State(x=time)


class ConstantIntegrator(ODENode):
    class State(NodeState):
        x: float = Var(init=0.0)

    def dstate(self, inputs: NodeInputs, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=1.0)


class CoupledSourceIntegrator(ODENode):
    class State(NodeState):
        x: float = Var(init=1.0)

    def dstate(self, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=0.0 * state.x)


class CoupledSinkIntegrator(ODENode):
    class Inputs(NodeInputs):
        source_x: float = Input(src=lambda: CoupledSourceIntegrator.State.x)

    class State(NodeState):
        x: float = Var(init=0.0)

    def dstate(self, inputs: Inputs) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=inputs.source_x)


def test_system_clock_is_readable_but_not_part_of_public_snapshot() -> None:
    counter = Counter()
    system = _one_phase_system(counter)

    assert system.read(Clock.tick) == 0
    assert system.read(Clock.time) == 0.0
    assert system.snapshot() == {"Counter.value": 0}

    system.step()

    assert system.read(Clock.tick) == 1
    assert system.read(Clock.time) == 1.0
    assert system.snapshot() == {"Counter.value": 1}


def test_system_clock_inputs_do_not_require_user_initial_state() -> None:
    logger = ClockLog()
    system = _one_phase_system(logger)

    assert system.compile_report.inputs["ClockLog.tick"] == "Clock.tick"
    assert system.compile_report.inputs["ClockLog.time"] == "Clock.time"
    assert "Clock.tick" not in system.compile_report.required_initial_state_vars
    assert "Clock.time" not in system.compile_report.required_initial_state_vars


def test_discrete_node_dt_runs_on_period_and_holds_state_between_updates() -> None:
    counter = Counter(dt=2)
    system = _one_phase_system(counter, base_dt=1)

    first_records = system.step()
    second_records = system.step()
    third_records = system.step()

    assert [record.node for record in first_records] == ["Counter"]
    assert second_records == ()
    assert [record.node for record in third_records] == ["Counter"]
    assert system.read("Counter.value") == 2
    assert system.read(Clock.tick) == 3
    assert system.read(Clock.time) == 3.0


def test_discrete_auto_base_dt_keeps_one_and_warns_about_idle_ticks() -> None:
    first = ScheduledCounter(name="first", dt=2)
    second = ScheduledCounter(name="second", dt=4)

    system = _one_phase_system(first, second)

    assert system.base_dt == Fraction(1, 1)
    assert any("idle ticks" in warning.message for warning in system.compile_report.warnings)


def test_explicit_base_dt_warns_when_schedule_has_idle_ticks() -> None:
    first = ScheduledCounter(name="first", dt=2)
    second = ScheduledCounter(name="second", dt=4)

    system = _one_phase_system(first, second, base_dt=1)

    assert any(
        "explicit base_dt=1 creates idle ticks" in warning.message
        for warning in system.compile_report.warnings
    )


def test_float_dt_is_rejected_for_discrete_and_ode_nodes() -> None:
    with pytest.raises(TypeError, match="must not be a float"):
        ScheduledCounter(dt=cast(Any, 0.1))

    with pytest.raises(TypeError, match="must not be a float"):
        ODESystem(nodes=(ConstantIntegrator(),), dt=cast(Any, 0.1))


def test_ode_node_rejects_instance_dt() -> None:
    with pytest.raises(TypeError, match="set dt on ODESystem"):
        ConstantIntegrator(dt="0.1")


def test_ode_node_rejects_class_dt() -> None:
    with pytest.raises(TypeError, match="set dt on ODESystem"):

        class BadIntegrator(ODENode):
            dt = "0.1"

            class State(NodeState):
                x: float = Var(init=0.0)

            def dstate(self, state: State) -> State:  # ty: ignore[invalid-method-override]
                return self.State(x=state.x)


def test_clock_is_reserved_node_name() -> None:
    with pytest.raises(CompileError) as exc_info:
        _one_phase_system(ScheduledCounter(name="Clock"))

    assert any("reserved system source" in issue.message for issue in exc_info.value.report.issues)


def test_phase_cannot_mix_ode_system_and_ordinary_nodes() -> None:
    ode = ODESystem(nodes=(ConstantIntegrator(),), dt="0.1")

    with pytest.raises(CompileError) as exc_info:
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "mixed",
                    nodes=(ode, ScheduledCounter()),
                    transitions=(Goto(terminate),),
                    is_initial=True,
                )
            ]
        )

    assert any("cannot mix ODESystem" in issue.message for issue in exc_info.value.report.issues)


def test_prs_allows_only_one_continuous_phase_for_now() -> None:
    first = ODESystem(nodes=(ConstantIntegrator(name="first"),), dt="0.1")
    second = ODESystem(nodes=(ConstantIntegrator(name="second"),), dt="0.1")

    with pytest.raises(CompileError) as exc_info:
        PhasedReactiveSystem(
            phases=[
                Phase("first", nodes=(first,), transitions=(Goto("second"),), is_initial=True),
                Phase("second", nodes=(second,), transitions=(Goto(terminate),)),
            ]
        )

    assert any("at most one phase" in issue.message for issue in exc_info.value.report.issues)


def test_independent_ode_systems_can_share_one_continuous_phase() -> None:
    first_ode = ODESystem(nodes=(ConstantIntegrator(name="first"),), dt="0.1")
    second_ode = ODESystem(nodes=(ConstantIntegrator(name="second"),), dt="0.1")

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "plant",
                nodes=(first_ode, second_ode),
                transitions=(Goto(terminate),),
                is_initial=True,
            )
        ]
    )

    system.step()

    assert system.read("first.x") == pytest.approx(0.1)
    assert system.read("second.x") == pytest.approx(0.1)


def test_continuous_phase_rejects_coupled_ode_systems() -> None:
    source_ode = ODESystem(nodes=(CoupledSourceIntegrator(),), dt="0.1")
    sink_ode = ODESystem(nodes=(CoupledSinkIntegrator(),), dt="0.1")

    with pytest.raises(CompileError) as exc_info:
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "plant",
                    nodes=(source_ode, sink_ode),
                    transitions=(Goto(terminate),),
                    is_initial=True,
                )
            ]
        )

    assert any(
        "Put continuously coupled ODENodes into the same ODESystem" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_ode_system_rejects_unlisted_ode_node_dependency() -> None:
    source = CoupledSourceIntegrator(name="source")

    class BoundSinkIntegrator(ODENode):
        class Inputs(NodeInputs):
            source_x: float = Input(src=source.State.x)

        class State(NodeState):
            x: float = Var(init=0.0)

        def dstate(self, inputs: Inputs) -> State:  # ty: ignore[invalid-method-override]
            return self.State(x=inputs.source_x)

    sink_ode = ODESystem(nodes=(BoundSinkIntegrator(name="sink"),), dt="0.1")

    with pytest.raises(CompileError) as exc_info:
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "plant",
                    nodes=(sink_ode,),
                    transitions=(Goto(terminate),),
                    is_initial=True,
                )
            ]
        )

    assert any(
        "source is not assigned to any phase" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_continuous_phase_commits_ode_state_and_updates_time_before_next_phase() -> None:
    plant = TimeDrivenIntegrator()
    ode = ODESystem(nodes=(plant,), dt="0.1")
    logger = ClockLog()

    system = PhasedReactiveSystem(
        phases=[
            Phase("plant", nodes=(ode,), transitions=(Goto("log"),), is_initial=True),
            Phase("log", nodes=(logger,), transitions=(Goto(terminate),)),
        ]
    )

    system.step()

    assert system.read("TimeDrivenIntegrator.x") == pytest.approx(0.005, rel=1e-3)
    assert system.read("ClockLog.times") == pytest.approx((0.1,))
    assert system.read("ClockLog.ticks") == (0,)
    assert system.read(Clock.time) == pytest.approx(0.1)
    assert system.read(Clock.tick) == 1


def test_reset_restores_continuous_node_internal_state() -> None:
    plant = ConstantIntegrator()
    ode = ODESystem(nodes=(plant,), dt="0.1")
    system = PhasedReactiveSystem(
        phases=[
            Phase("plant", nodes=(ode,), transitions=(Goto(terminate),), is_initial=True),
        ]
    )

    system.step()
    assert system.read("ConstantIntegrator.x") == pytest.approx(0.1)

    system.reset(initial_state={"ConstantIntegrator.x": 2.0})

    assert system.read("ConstantIntegrator.x") == pytest.approx(2.0)
    assert cast(ConstantIntegrator.State, plant.state()).x == pytest.approx(2.0)
    assert system.read(Clock.time) == 0.0


def test_clock_can_drive_transition_guards() -> None:
    early = ScheduledCounter(name="early")
    late = ScheduledCounter(name="late")

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "start",
                nodes=(),
                transitions=(
                    If(V(Clock.tick) >= 1, "late"),
                    Else("early"),
                ),
                is_initial=True,
            ),
            Phase("early", nodes=(early,), transitions=(Goto(terminate),)),
            Phase("late", nodes=(late,), transitions=(Goto(terminate),)),
        ]
    )

    system.step()
    system.step()

    assert system.read("early.value") == 1
    assert system.read("late.value") == 1
