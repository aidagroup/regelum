from __future__ import annotations

from fractions import Fraction
from typing import Any, cast

import pytest

from regelum import (
    Clock,
    CompileError,
    Else,
    ElseIf,
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


def test_ode_system_dt_contributes_to_auto_base_dt_but_not_runtime_skip() -> None:
    first_ode = ODESystem(nodes=(ConstantIntegrator(name="first"),), dt="0.1")
    second_ode = ODESystem(nodes=(ConstantIntegrator(name="second"),), dt="0.2")

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "plant",
                nodes=(first_ode, second_ode),
                transitions=(Goto(terminate),),
                is_initial=True,
            )
        ],
    )

    assert system.base_dt == Fraction(1, 10)

    first_records = system.step()
    second_records = system.step()

    assert [record.node for record in first_records] == ["ODESystem", "ODESystem"]
    assert [record.node for record in second_records] == ["ODESystem", "ODESystem"]
    assert system.read("first.x") == pytest.approx(0.2)
    assert system.read("second.x") == pytest.approx(0.2)
    assert system.read(Clock.time) == pytest.approx(0.2)


def test_ode_integrates_on_base_dt_not_ode_dt_when_base_dt_is_smaller() -> None:
    ode = ODESystem(nodes=(ConstantIntegrator(),), dt="0.1")
    system = PhasedReactiveSystem(
        phases=[
            Phase("plant", nodes=(ode,), transitions=(Goto(terminate),), is_initial=True),
        ],
        base_dt="0.01",
    )

    system.run(steps=3)

    assert system.base_dt == Fraction(1, 100)
    assert len(system.history) == 3
    assert system.read("ConstantIntegrator.x") == pytest.approx(0.03)
    assert system.read(Clock.time) == pytest.approx(0.03)


def test_discrete_dt_samples_and_holds_while_ode_integrates_every_base_tick() -> None:
    class Controller(Node):
        def __init__(
            self,
            *,
            name: str | None = None,
            dt: Fraction | int | str | None = None,
        ) -> None:
            super().__init__(name=name, dt=dt)
            self.count = 0

        class State(NodeState):
            value: int = Var(init=0)

        def update(self) -> State:
            self.count += 1
            return self.State(value=self.count)

    class DrivenIntegrator(ODENode):
        class Inputs(NodeInputs):
            source_x: float = Input(src=Controller.State.value)

        class State(NodeState):
            x: float = Var(init=0.0)

        def dstate(self, inputs: Inputs) -> State:  # ty: ignore[invalid-method-override]
            return self.State(x=inputs.source_x)

    controller = Controller(dt="0.2")
    plant = DrivenIntegrator(name="plant")
    ode = ODESystem(nodes=(plant,), dt="0.1")

    system = PhasedReactiveSystem(
        phases=[
            Phase("control", nodes=(controller,), transitions=(Goto("plant"),), is_initial=True),
            Phase("plant", nodes=(ode,), transitions=(Goto(terminate),)),
        ],
        connections=(plant.Inputs.source_x.connect(controller.State.value),),
        base_dt="0.1",
    )

    system.run(steps=3)

    assert controller.count == 2
    assert system.read("Controller.value") == 2
    assert system.read("plant.x") == pytest.approx(0.4)
    assert system.read(Clock.time) == pytest.approx(0.3)


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


def test_compile_rejects_feasible_path_that_skips_continuous_phase() -> None:
    class Mode(Node):
        class State(NodeState):
            bypass: bool = Var(init=False)

        def update(self) -> State:
            return self.State(bypass=False)

    mode = Mode()
    ode = ODESystem(nodes=(ConstantIntegrator(),), dt="0.1")

    with pytest.raises(CompileError) as exc_info:
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "select",
                    nodes=(mode,),
                    transitions=(
                        If(V(mode.State.bypass), terminate, name="bypass"),
                        Else("plant", name="integrate"),
                    ),
                    is_initial=True,
                ),
                Phase("plant", nodes=(ode,), transitions=(Goto(terminate),)),
            ],
        )

    assert any(
        "continuous phase contract violation" in issue.message
        and "without reaching a continuous phase" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_compile_accepts_symbolically_infeasible_path_that_skips_continuous_phase() -> None:
    class Mode(Node):
        class State(NodeState):
            bypass: bool = Var(init=False)

        def update(self) -> State:
            return self.State(bypass=False)

    mode = Mode()
    ode = ODESystem(nodes=(ConstantIntegrator(),), dt="0.1")

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "select",
                nodes=(mode,),
                transitions=(
                    If(V(mode.State.bypass) & ~V(mode.State.bypass), terminate, name="bypass"),
                    Else("plant", name="integrate"),
                ),
                is_initial=True,
            ),
            Phase("plant", nodes=(ode,), transitions=(Goto(terminate),)),
        ],
    )

    assert system.compile_report.ok


def test_compile_rejects_feasible_path_that_reaches_continuous_phase_twice() -> None:
    class Mode(Node):
        class State(NodeState):
            repeat: bool = Var(init=False)

        def update(self) -> State:
            return self.State(repeat=False)

    mode = Mode()
    ode = ODESystem(nodes=(ConstantIntegrator(),), dt="0.1")

    with pytest.raises(CompileError) as exc_info:
        PhasedReactiveSystem(
            phases=[
                Phase("plant", nodes=(ode,), transitions=(Goto("select"),), is_initial=True),
                Phase(
                    "select",
                    nodes=(mode,),
                    transitions=(
                        If(V(mode.State.repeat), "plant", name="repeat"),
                        Else(terminate, name="done"),
                    ),
                ),
            ],
        )

    assert any(
        "continuous phase contract violation" in issue.message
        and "more than once" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_continuous_contract_accepts_many_symbolic_branches_that_all_rejoin_before_plant() -> None:
    class Router(Node):
        class State(NodeState):
            mode: int = Var(init=0, domain=(0, 1, 2))
            armed: bool = Var(init=True)

        def update(self) -> State:
            return self.State(mode=0, armed=True)

    router = Router()
    ode = ODESystem(nodes=(ConstantIntegrator(),), dt="0.1")

    system = PhasedReactiveSystem(
        phases=[
            Phase(
                "route",
                nodes=(router,),
                transitions=(
                    If(V(router.State.mode) == 0, "left", name="left"),
                    ElseIf(V(router.State.mode) == 1, "middle", name="middle"),
                    Else("right", name="right"),
                ),
                is_initial=True,
            ),
            Phase(
                "left",
                nodes=(),
                transitions=(
                    If(V(router.State.armed), "plant", name="armed"),
                    Else("plant", name="not-armed"),
                ),
            ),
            Phase("middle", nodes=(), transitions=(Goto("plant"),)),
            Phase("right", nodes=(), transitions=(Goto("plant"),)),
            Phase("plant", nodes=(ode,), transitions=(Goto(terminate),)),
        ],
    )

    assert system.compile_report.ok


def test_continuous_contract_accepts_symbolic_branches_after_plant_that_all_terminate() -> None:
    class Router(Node):
        class State(NodeState):
            mode: int = Var(init=0, domain=(0, 1, 2))
            armed: bool = Var(init=True)

        def update(self) -> State:
            return self.State(mode=0, armed=True)

    router = Router()
    ode = ODESystem(nodes=(ConstantIntegrator(),), dt="0.1")

    system = PhasedReactiveSystem(
        phases=[
            Phase("plant", nodes=(ode,), transitions=(Goto("route"),), is_initial=True),
            Phase(
                "route",
                nodes=(router,),
                transitions=(
                    If((V(router.State.mode) == 0) & V(router.State.armed), "a", name="a"),
                    ElseIf(V(router.State.mode) == 1, "b", name="b"),
                    Else("c", name="c"),
                ),
            ),
            Phase("a", nodes=(), transitions=(Goto(terminate),)),
            Phase("b", nodes=(), transitions=(Goto(terminate),)),
            Phase("c", nodes=(), transitions=(Goto(terminate),)),
        ],
    )

    assert system.compile_report.ok


def test_continuous_contract_accepts_infeasible_second_plant_branch() -> None:
    class Router(Node):
        class State(NodeState):
            repeat: bool = Var(init=False)

        def update(self) -> State:
            return self.State(repeat=False)

    router = Router()
    ode = ODESystem(nodes=(ConstantIntegrator(),), dt="0.1")

    system = PhasedReactiveSystem(
        phases=[
            Phase("prepare", nodes=(router,), transitions=(Goto("plant"),), is_initial=True),
            Phase("plant", nodes=(ode,), transitions=(Goto("route"),)),
            Phase(
                "route",
                nodes=(),
                transitions=(
                    If(V(router.State.repeat) & ~V(router.State.repeat), "plant", name="repeat"),
                    Else(terminate, name="done"),
                ),
            ),
        ]
    )

    assert system.compile_report.ok


def test_continuous_contract_rejects_havoced_guard_that_can_skip_plant() -> None:
    class Router(Node):
        class State(NodeState):
            bypass: bool = Var(init=False)

        def update(self) -> State:
            return self.State(bypass=False)

    router = Router()
    ode = ODESystem(nodes=(ConstantIntegrator(),), dt="0.1")

    with pytest.raises(CompileError) as exc_info:
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "route",
                    nodes=(router,),
                    transitions=(
                        If(V(router.State.bypass), terminate, name="bypass"),
                        Else("plant", name="plant"),
                    ),
                    is_initial=True,
                ),
                Phase("plant", nodes=(ode,), transitions=(Goto(terminate),)),
            ],
        )

    assert any(
        issue.location == "route"
        and "without reaching a continuous phase" in issue.message
        and "Router.bypass" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_continuous_contract_rejects_non_symbolic_guard_before_mandatory_plant() -> None:
    ode = ODESystem(nodes=(ConstantIntegrator(),), dt="0.1")

    with pytest.raises(CompileError) as exc_info:
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "route",
                    nodes=(),
                    transitions=(
                        If(lambda _state: True, "plant", name="python-guard"),
                        Else(terminate, name="fallback"),
                    ),
                    is_initial=True,
                ),
                Phase("plant", nodes=(ode,), transitions=(Goto(terminate),)),
            ],
        )

    assert any(
        issue.location == "route.python-guard"
        and "non-symbolic transition guards" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_continuous_contract_rejects_deep_branch_that_can_skip_plant_after_rejoin() -> None:
    class Router(Node):
        class State(NodeState):
            a: bool = Var(init=False)
            b: bool = Var(init=False)
            c: bool = Var(init=False)

        def update(self) -> State:
            return self.State(a=False, b=False, c=False)

    router = Router()
    ode = ODESystem(nodes=(ConstantIntegrator(),), dt="0.1")

    with pytest.raises(CompileError) as exc_info:
        PhasedReactiveSystem(
            phases=[
                Phase(
                    "a",
                    nodes=(router,),
                    transitions=(
                        If(V(router.State.a), "b1", name="b1"),
                        Else("b2", name="b2"),
                    ),
                    is_initial=True,
                ),
                Phase(
                    "b1",
                    nodes=(),
                    transitions=(
                        If(V(router.State.b), "c1", name="c1"),
                        Else("c2", name="c2"),
                    ),
                ),
                Phase("b2", nodes=(), transitions=(Goto("c2"),)),
                Phase(
                    "c1",
                    nodes=(),
                    transitions=(
                        If(V(router.State.c), "plant", name="plant"),
                        Else(terminate, name="bad"),
                    ),
                ),
                Phase("c2", nodes=(), transitions=(Goto("plant"),)),
                Phase("plant", nodes=(ode,), transitions=(Goto(terminate),)),
            ],
        )

    assert any(
        "a -> b1 -> c1" in issue.location
        and "without reaching a continuous phase" in issue.message
        for issue in exc_info.value.report.issues
    )


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
