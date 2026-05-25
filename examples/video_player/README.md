# Adaptive-Bitrate Video Player

This example models an adaptive-bitrate video player as a discrete feedback
loop. Network bandwidth changes over time, a quality policy predicts whether
the buffer is about to stall, and a bitrate controller drops the target quality
when needed before playback consumes the next chunk.

Regelum is built for feedback systems where each step commits state that shapes
later behavior. That pattern appears in closed-loop control, robotics, power
systems, autonomous driving, and industrial automation. Before simulation,
Regelum compiles the declarative PRS and checks that phases can be scheduled,
guarded transitions choose a single successor where they are symbolic, and each
tick reaches termination. This example makes the guarded branch visible:
`decide` either jumps to `drop_quality` or directly to `play`.

Documentation: [phases](../../docs/concepts/phases.md) and
[create, compile, and run](../../docs/concepts/compilation.md).

## What It Shows

- `Network` publishes deterministic bandwidth changes.
- `QualityPolicy` computes a boolean `stalling` state.
- `BitrateController` lowers the bitrate ladder only on the stalling branch.
- `Decoder`, `MediaSession`, and `Logger` run in the `play` phase.
- The phase graph is `measure -> decide -> play` or
  `measure -> decide -> drop_quality -> play`.

## What It Displays

The script prints compile status, the resolved phase schedules, and a 30-tick
table with bandwidth, bitrate, buffer level, stalling state, and the phase path
taken in each tick.

Run it:

```bash
uv run python examples/video_player/video_player.py
```
