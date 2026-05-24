import pytest

import regelum as rg
from examples.compile_checks.complex_safety_loop import build_bad_c1_system, build_ok_system


def test_complex_safety_loop_ok_example_runs() -> None:
    system = build_ok_system()

    assert system.compile_report.ok
    system.run(steps=3)
    assert len(system.snapshot()["TraceLogger.trace"]) == 3


def test_complex_safety_loop_bad_example_fails_c1() -> None:
    with pytest.raises(rg.CompileError) as exc_info:
        build_bad_c1_system()

    assert any(
        issue.location == "bad-coupled-control" and issue.message.startswith("C1 violation")
        for issue in exc_info.value.report.issues
    )
