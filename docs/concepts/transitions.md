# Transitions

Transitions choose the next phase after the current phase has executed.
They may also terminate the current tick.

The video player uses three transition styles: `Goto` for the linear segments,
`If`/`Else` for the branching decision, and `Goto(terminate)` to end the
tick.

## Goto

`Goto` is an unconditional transition.
`measure`, `drop_quality`, and `play` end with one.

```python
rg.Phase(
    "measure",
    nodes=(clock, network),
    transitions=(rg.Goto("decide"),),
    is_initial=True,
)

rg.Phase(
    "play",
    nodes=(decoder, session, logger),
    transitions=(rg.Goto(rg.terminate),),
)
```

`rg.terminate` ends the current tick.

## Branch chains

Use `If`, `Elif`, and `Else` for ordered branching.
The video player only needs `If` plus `Else`, because the policy publishes a
single boolean:

```python
rg.Phase(
    "decide",
    nodes=(policy,),
    transitions=(
        rg.If(rg.V(policy.Outputs.stalling), "drop_quality", name="stalling"),
        rg.Else("play", name="healthy"),
    ),
)
```

A larger system might extend the chain with `Elif`:

```python
transitions = (
    rg.If(rg.V(policy.Outputs.stalling), "drop_quality"),
    rg.Elif(rg.V(policy.Outputs.healthy_steady), "upgrade_quality"),
    rg.Else("play"),
)
```

`ElseIf` is also available.
`Elif` and `ElseIf` behave the same.
An `Else` closes the current branch chain; `Elif` after `Else` is invalid.

## Multiple chains

A phase may contain several independent `If` chains.
Compilation checks that the transition structure is well formed.
Runtime evaluates effective transitions in order.

## Symbolic predicates

Use `V(...)` to read outputs in guards.

```python
rg.If(rg.V(policy.Outputs.stalling), "drop_quality")
rg.If(rg.V(MediaSession.Outputs.buffer_seconds) < 2.0, "buffer_warning")
```

`V(...)` accepts the same kinds of output references as inputs:

- output descriptors;
- instance-bound output ports;
- lazy callables;
- string references.

## Enum values

Enum outputs can be compared directly.

```python
from enum import Enum


class PlaybackMode(Enum):
    PLAYING = "playing"
    PAUSED = "paused"


rg.If(rg.V(state.Outputs.mode) == PlaybackMode.PLAYING, "play")
```

## Python predicates

Python callables can be used as predicates.
They are allowed as an escape hatch, but symbolic predicates are preferred —
they enable C3 verification and clearer compile diagnostics.

## Rules

- Use `Goto` for unconditional jumps.
- Use `If` / `Elif` / `Else` for ordered branches.
- Use `rg.terminate` to end a tick.
- Prefer symbolic `V(...)` predicates.
- Keep guard producers covered by the phase graph.
