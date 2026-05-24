from examples.instance_connect.instance_connect import build_system


def test_instance_connect_example_runs() -> None:
    system = build_system()

    system.run(steps=3)

    snapshot = system.snapshot()
    assert snapshot["accumulator_a.total"] == 9
    assert snapshot["accumulator_b.total"] == 21
