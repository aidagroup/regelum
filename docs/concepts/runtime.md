# Runtime

Runtime executes compiled phases and updates system state.
The runtime does not reinterpret declarations on every step.

## The tick

A tick walks the phase graph from the initial phase until a transition reaches
`terminate`.
For the video player, a healthy tick visits `measure → decide → play`; a
stalling tick visits `measure → decide → drop_quality → play`.

The feedback loop closes between ticks: `MediaSession.buffer_seconds` written
in `play` of tick N is read by `QualityPolicy` in `decide` of tick N+1, and
that read is what selects the branch.

## Step order

One `step()` starts at the initial phase and follows transitions until the
tick terminates.

For each phase:

1. run active nodes in the compiled schedule;
2. build each node input namespace;
3. call `run`;
4. normalize returned outputs;
5. write outputs into state;
6. choose the next phase from transitions.

```python
records = system.step()
```

Each record contains the phase, node, inputs, and outputs.
A 30-tick run of the player produces records like:

```python
for record in records:
    print(record.phase, record.node, record.outputs)
# measure Clock {'tick': 7}
# measure Network {'bandwidth_kbps': 600.0}
# decide  QualityPolicy {'stalling': False}
# play    Decoder {'fetched_seconds': 0.278}
# play    MediaSession {'buffer_seconds': 9.11}
# play    Logger {'history': [...]}
```

## Running multiple ticks

Use `run(steps=...)` to execute several ticks.

```python
system.run(steps=30)
```

Each tick starts from the initial phase again.
State persists across ticks unless `reset()` is called.

## State access

Use `snapshot()` to inspect current state.

```python
snapshot = system.snapshot()
print(snapshot["MediaSession.buffer_seconds"])
print(snapshot["BitrateController.value"])
```

Use `read(...)` when code has an output reference.

```python
buffer = system.read(session.Outputs.buffer_seconds)
```

## Reset

`reset()` clears runtime state and history.
It then applies declared initial values and optional overrides.

```python
system.reset()
system.reset(initial_state={MediaSession.Outputs.buffer_seconds: 5.0})
```

## Logging nodes

A logger is just another node.
It sees the values available at the point where its scheduled phase runs.
The video player puts `Logger` last in `play` so it observes the buffer
update from `MediaSession` and the freshly committed bitrate.

## Rules

- Runtime follows compiled phase schedules.
- State persists between ticks.
- `reset()` clears state and history.
- `step()` returns execution records.
- `snapshot()` returns a copy of the current state mapping.
