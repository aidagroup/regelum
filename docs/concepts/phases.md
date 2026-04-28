# Phases

A phase selects the node instances that are active together.
Ticks move through one or more phases until a transition terminates.

The video player has four phases — `measure`, `decide`, `drop_quality`, and
`play`.
`measure` and `play` carry several nodes each; `decide` and `drop_quality`
each hold one.

## Phase nodes

`Phase.nodes` accepts node instances only.
Node classes are rejected immediately.

```python
rg.Phase(
    "play",
    nodes=(decoder, session, logger),
    transitions=(rg.Goto(rg.terminate),),
)
```

The `PhasedReactiveSystem` node set is derived from phases.
There is no separate `nodes=[...]` list on the system.

## Initial phase

Exactly one phase should be marked `is_initial=True`.
There is no fallback to the first phase.

```python
rg.Phase(
    "measure",
    nodes=(clock, network),
    transitions=(rg.Goto("decide"),),
    is_initial=True,
)
```

The initial phase is *where every tick enters the cycle*.
For the player, that is `measure`: a tick begins by sampling the clock and the
network before any decision is taken.

## Complete phase graph

A phase must cover the nodes needed by its inputs and guards.
The compiler walks the dependency graph from phase nodes and guard references.
If a producer is outside all phase coverage, compilation reports an incomplete
phase graph.

For the player, `decide` reads `MediaSession.buffer_seconds`,
`BitrateController.value`, and `Network.bandwidth_kbps` from committed state.
All three producers belong to other phases, which is fine — the constraint
applies to the union of phases, not each phase in isolation.

## Phase schedule

Nodes inside a phase run in topological order.
Input dependencies define the order; the position in `Phase.nodes` only
breaks ties between independent nodes.

For the player:

| Phase | Order | Notes |
|---|---|---|
| `measure` | `Clock`, `Network` | `Network` reads `Clock.tick`, so `Clock` precedes it. |
| `decide` | `QualityPolicy` | Single node. |
| `drop_quality` | `BitrateController` | Single node. |
| `play` | `Decoder`, `MediaSession`, `Logger` | `MediaSession` reads `Decoder.fetched_seconds`; `Logger` reads everything else. |

```python
print(system.compile_report.phase_schedules)
print(system.compile_report.phase_dependency_edges)
```

## Rules

- Phases list node instances, not classes.
- A system is defined by its phases.
- Exactly one initial phase is expected.
- Each phase is scheduled topologically.
- Phase coverage must include input and guard producers.
