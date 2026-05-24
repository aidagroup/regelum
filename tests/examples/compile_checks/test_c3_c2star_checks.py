import pytest

import regelum as rg
from examples.compile_checks.c3_c2star_checks import (
    build_c2star_system,
    build_c3_violation,
    main,
)


def test_c3_c2star_checks_example_status(capsys: pytest.CaptureFixture[str]) -> None:
    main()

    assert capsys.readouterr().out.splitlines() == [
        "phi: C3 violation: transitions 'if' and 'if' overlap at state {'X.x': 'False'}",
        "compile ok = True",
        "C2*(2) status = pass",
    ]


def test_c3_c2star_checks_example_contracts() -> None:
    with pytest.raises(rg.CompileError):
        build_c3_violation()

    assert build_c2star_system().compile_report.ok
