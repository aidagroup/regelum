import pytest

import regelum as rg
from examples.compile_checks.unconditional_cycle import build_system


def test_unconditional_cycle_example_fails_c2star() -> None:
    with pytest.raises(rg.CompileError) as exc_info:
        build_system()

    assert any(
        issue.location == "SCC {a, b}"
        and issue.message
        == "C2*: SAT at finite bound N_S=2; infinite local residence is possible, but global nontermination requires a reachable entry"
        for issue in exc_info.value.report.issues
    )
