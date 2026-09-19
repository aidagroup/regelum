import pytest

import regelum as rg
from examples.compile_checks.c2star_cycle import build_dead_cycle_system, build_live_cycle_system


def test_dead_cycle_preserves_guard_variable_and_compiles():
    assert build_dead_cycle_system().compile_report.ok


def test_live_cycle_can_rewrite_guard_variable_and_is_rejected():
    with pytest.raises(rg.CompileError) as exc:
        build_live_cycle_system()
    assert any("SAT at finite bound N_S=4" in i.message for i in exc.value.report.issues)
