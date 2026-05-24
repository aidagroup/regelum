import pytest

import regelum as rg
from examples.compile_checks.complex_c3_partition import build_bad_overlap_system, build_ok_system


def test_complex_c3_ok_example_compiles() -> None:
    assert build_ok_system().compile_report.ok


def test_complex_c3_bad_example_fails_partition() -> None:
    with pytest.raises(rg.CompileError) as exc_info:
        build_bad_overlap_system()

    assert any(
        issue.location == "diagnose" and issue.message.startswith("C3 violation")
        for issue in exc_info.value.report.issues
    )
