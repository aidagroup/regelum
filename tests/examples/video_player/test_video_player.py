from examples.video_player.video_player import build_system


def test_video_player_example_runs() -> None:
    system = build_system()

    system.run(steps=3)

    assert len(system.snapshot()["Logger.history"]) == 3
