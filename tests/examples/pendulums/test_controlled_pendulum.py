from examples.controlled_pendulum.standalone import run


def test_controlled_pendulum_example_runs() -> None:
    samples = run(steps=5)

    assert len(samples) == 5
    assert abs(samples[-1][3]) <= 4.0
