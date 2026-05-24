import pytest

from examples.free_pendulum.standalone import build_system


def test_free_pendulum_example_runs_and_logs_samples() -> None:
    system = build_system()

    system.run(steps=5)

    samples = system.snapshot()["Logger.samples"]
    assert len(samples) == 5
    assert samples[-1][0] == pytest.approx(0.05)
