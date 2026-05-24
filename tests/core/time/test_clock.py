import pytest

import regelum as rg
from tests.core.time._support import (
    ClockLog,
    Counter,
    ScheduledCounter,
    _one_phase_system,
)


def test_system_clock_is_readable_but_not_part_of_public_snapshot() -> None:
    counter = Counter()
    system = _one_phase_system(counter)

    assert system.read(rg.Clock.tick) == 0
    assert system.read(rg.Clock.time) == 0.0
    assert system.snapshot() == {"Counter.value": 0}

    system.step()

    assert system.read(rg.Clock.tick) == 1
    assert system.read(rg.Clock.time) == 1.0
    assert system.snapshot() == {"Counter.value": 1}


def test_system_clock_inputs_do_not_require_user_initial_state() -> None:
    logger = ClockLog()
    system = _one_phase_system(logger)

    assert system.compile_report.inputs["ClockLog.tick"] == "Clock.tick"
    assert system.compile_report.inputs["ClockLog.time"] == "Clock.time"
    assert "Clock.tick" not in system.compile_report.required_initial_state_vars
    assert "Clock.time" not in system.compile_report.required_initial_state_vars


def test_clock_is_reserved_node_name() -> None:
    with pytest.raises(rg.CompileError) as exc_info:
        _one_phase_system(ScheduledCounter(name="Clock"))

    assert any("reserved system source" in issue.message for issue in exc_info.value.report.issues)


def test_clock_can_drive_transition_guards() -> None:
    early = ScheduledCounter(name="early")
    late = ScheduledCounter(name="late")

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "start",
                nodes=(),
                transitions=(
                    rg.If(rg.V(rg.Clock.tick) >= 1, "late"),
                    rg.Else("early"),
                ),
                is_initial=True,
            ),
            rg.Phase("early", nodes=(early,), transitions=(rg.Goto(rg.terminate),)),
            rg.Phase("late", nodes=(late,), transitions=(rg.Goto(rg.terminate),)),
        ]
    )

    system.step()
    system.step()

    assert system.read("early.value") == 1
    assert system.read("late.value") == 1
