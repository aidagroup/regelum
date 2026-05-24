import pytest

import regelum as rg
from examples.compile_checks.c1_violation import build_system


def test_c1_violation_example_fails_c1() -> None:
    with pytest.raises(rg.CompileError) as exc_info:
        build_system()

    assert any(
        issue.location == "coupled" and issue.message.startswith("C1 violation")
        for issue in exc_info.value.report.issues
    )
