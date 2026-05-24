import pytest

import regelum as rg
from examples.compile_checks.c3_violation import build_system


def test_c3_violation_example_fails_c3() -> None:
    with pytest.raises(rg.CompileError) as exc_info:
        build_system()

    assert any(
        issue.location == "phi" and issue.message.startswith("C3 violation")
        for issue in exc_info.value.report.issues
    )
