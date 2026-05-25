import pytest

import regelum as rg
from examples.compile_checks.unconditional_cycle import build_system


def test_unconditional_cycle_example_fails_c2star() -> None:
    with pytest.raises(rg.CompileError) as exc_info:
        build_system()

    assert any(
        issue.location == "a -> b -> a"
        and issue.message
        == "C2*(1) violation: cycle is feasible for 1 traversal(s), R_C=[], witness={}"
        for issue in exc_info.value.report.issues
    )
