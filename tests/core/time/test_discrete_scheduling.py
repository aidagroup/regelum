from typing import cast

import pytest

import regelum as rg
from tests.core.time._support import (
    Any,
    ConstantIntegrator,
    Counter,
    Fraction,
    ScheduledCounter,
    _one_phase_system,
)


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
    assert system.read(rg.Clock.tick) == 3
    assert system.read(rg.Clock.time) == 3.0


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


def test_discrete_node_accepts_class_dt() -> None:
    class SlowCounter(rg.Node):
        dt = "2"

        def __init__(self) -> None:
            self.count = 0

        class State(rg.NodeState):
            value: int = rg.var(init=0)

        def update(self) -> State:
            self.count += 1
            return self.State(value=self.count)

    counter = SlowCounter()
    system = _one_phase_system(counter, base_dt=1)

    system.run(steps=3)

    assert counter.count == 2


def test_float_dt_is_rejected_for_discrete_and_ode_nodes() -> None:
    with pytest.raises(TypeError, match="must not be a float"):
        ScheduledCounter(dt=cast(Any, 0.1))

    with pytest.raises(TypeError, match="must not be a float"):
        rg.ODESystem(nodes=(ConstantIntegrator(),), dt=cast(Any, 0.1))
