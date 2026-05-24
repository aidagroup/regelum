from examples.compile_checks.c2star_split_writes import build_system


def test_c2star_split_writes_example_compiles() -> None:
    system = build_system(p_x=0.0, p_y=1.0, seed=0)

    assert system.compile_report.ok
