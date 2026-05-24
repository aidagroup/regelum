from typing import cast

import pytest

import regelum as rg
from tests.core.time._support import (
    ClockLog,
    ConstantIntegrator,
    Fraction,
    TimeDrivenIntegrator,
)


def test_ode_system_dt_contributes_to_auto_base_dt_but_not_runtime_skip() -> None:
    first_ode = rg.ODESystem(nodes=(ConstantIntegrator(name="first"),), dt="0.1")
    second_ode = rg.ODESystem(nodes=(ConstantIntegrator(name="second"),), dt="0.2")

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "plant",
                nodes=(first_ode, second_ode),
                transitions=(rg.Goto(rg.terminate),),
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
    assert system.read(rg.Clock.time) == pytest.approx(0.2)


def test_ode_integrates_on_base_dt_not_ode_dt_when_base_dt_is_smaller() -> None:
    ode = rg.ODESystem(nodes=(ConstantIntegrator(),), dt="0.1")
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase("plant", nodes=(ode,), transitions=(rg.Goto(rg.terminate),), is_initial=True),
        ],
        base_dt="0.01",
    )

    system.run(steps=3)

    assert system.base_dt == Fraction(1, 100)
    assert len(system.history) == 3
    assert system.read("ConstantIntegrator.x") == pytest.approx(0.03)
    assert system.read(rg.Clock.time) == pytest.approx(0.03)


def test_discrete_dt_samples_and_holds_while_ode_integrates_every_base_tick() -> None:
    class Controller(rg.Node):
        def __init__(
            self,
            *,
            name: str | None = None,
            dt: Fraction | int | str | None = None,
        ) -> None:
            super().__init__(name=name, dt=dt)
            self.count = 0

        class State(rg.NodeState):
            value: int = rg.var(init=0)

        def update(self) -> State:
            self.count += 1
            return self.State(value=self.count)

    class DrivenIntegrator(rg.ODENode):
        class Inputs(rg.NodeInputs):
            source_x: float = rg.src(Controller.State.value)

        class State(rg.NodeState):
            x: float = rg.var(init=0.0)

        def dstate(self, inputs: Inputs) -> State:  # ty: ignore[invalid-method-override]
            return self.State(x=inputs.source_x)

    controller = Controller(dt="0.2")
    plant = DrivenIntegrator(name="plant")
    ode = rg.ODESystem(nodes=(plant,), dt="0.1")

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "control", nodes=(controller,), transitions=(rg.Goto("plant"),), is_initial=True
            ),
            rg.Phase("plant", nodes=(ode,), transitions=(rg.Goto(rg.terminate),)),
        ],
        connections=(plant.Inputs.source_x.connect(controller.State.value),),
        base_dt="0.1",
    )

    system.run(steps=3)

    assert controller.count == 2
    assert system.read("Controller.value") == 2
    assert system.read("plant.x") == pytest.approx(0.4)
    assert system.read(rg.Clock.time) == pytest.approx(0.3)


def test_continuous_phase_commits_ode_state_and_updates_time_before_next_phase() -> None:
    plant = TimeDrivenIntegrator()
    ode = rg.ODESystem(nodes=(plant,), dt="0.1")
    logger = ClockLog()

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase("plant", nodes=(ode,), transitions=(rg.Goto("log"),), is_initial=True),
            rg.Phase("log", nodes=(logger,), transitions=(rg.Goto(rg.terminate),)),
        ]
    )

    system.step()

    assert system.read("TimeDrivenIntegrator.x") == pytest.approx(0.005, rel=1e-3)
    assert system.read("ClockLog.times") == pytest.approx((0.1,))
    assert system.read("ClockLog.ticks") == (0,)
    assert system.read(rg.Clock.time) == pytest.approx(0.1)
    assert system.read(rg.Clock.tick) == 1


def test_reset_restores_continuous_node_internal_state() -> None:
    plant = ConstantIntegrator()
    ode = rg.ODESystem(nodes=(plant,), dt="0.1")
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase("plant", nodes=(ode,), transitions=(rg.Goto(rg.terminate),), is_initial=True),
        ]
    )

    system.step()
    assert system.read("ConstantIntegrator.x") == pytest.approx(0.1)

    system.reset(initial_state={"ConstantIntegrator.x": 2.0})

    assert system.read("ConstantIntegrator.x") == pytest.approx(2.0)
    assert cast(ConstantIntegrator.State, plant.state()).x == pytest.approx(2.0)
    assert system.read(rg.Clock.time) == 0.0
