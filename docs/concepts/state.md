# State and Initialization

System state stores output values.
Inputs read state values or values produced earlier in the same phase.
The closed loop in the video player exists *only* because state persists
between ticks: `MediaSession.buffer_seconds` written at the end of tick N is
what `QualityPolicy` reads in `decide` of tick N+1.

## Initial values

Outputs that are read before they are produced need initial values.

```python
class MediaSession(rg.Node):
    class Outputs(rg.NodeOutputs):
        buffer_seconds: float = rg.Output(initial=10.0)
```

`buffer_seconds` is read by `QualityPolicy` in the `decide` phase, *before*
`MediaSession` writes it again in the `play` phase, so an `initial` value is
required.
The same applies to `Clock.tick`, `Network.bandwidth_kbps`,
`BitrateController.value`, and `QualityPolicy.stalling`.

Outputs that are produced before their first read may omit `initial`.

```python
class Decoder(rg.Node):
    class Outputs(rg.NodeOutputs):
        fetched_seconds: float
```

`Decoder.fetched_seconds` is fresh in every `play` phase before
`MediaSession` reads it in the same phase, so no initial value is needed.

Compilation computes the minimal initial state needed for the system.

```python
report = system.compile_report
print(report.minimal_initial_outputs)
print(report.required_initial_outputs)
```

For the video player, `minimal_initial_outputs` lists the five outputs that
must persist across ticks.

## Static and callable initial values

Static initial values are direct.

```python
buffer_seconds: float = rg.Output(initial=10.0)
```

Use a zero-argument callable for fresh containers.
The video player's `Logger.history` uses this idiom so each new system gets
its own list:

```python
class Logger(rg.Node):
    class Outputs(rg.NodeOutputs):
        history: list[Logger.Sample] = rg.Output(initial=lambda: [])
```

Use a one-argument callable for instance-dependent configuration.
The argument name may be `self`; it is still just the node instance passed by
the runtime.

```python
buffer_seconds: float = rg.Output(
    initial=lambda self: cast(MediaSession, self).initial_buffer,
)
```

## Tick-zero overrides

`initial_state` overrides initial values for a single run.
It does not mutate node declarations.
This is useful for sweeping starting buffer levels or starting bitrates
without rebuilding the system.

```python
system = rg.PhasedReactiveSystem(
    phases=phases,
    initial_state={MediaSession.Outputs.buffer_seconds: 2.0},
)

system.reset(
    initial_state={
        MediaSession.Outputs.buffer_seconds: 5.0,
        BitrateController.Outputs.value: 720,
    }
)
```

## Object identity

State stores Python objects by reference.
Use callable initial values for mutable containers (the `history` example
above).
Return the same object when intentional in-place enrichment is desired —
`Logger` does this, appending to `inputs.history` and returning the same
list.

```python
def run(self, inputs: Inputs) -> Outputs:
    inputs.history.append(record)
    return self.Outputs(history=inputs.history)
```

## Rules

- Outputs define state.
- Previous-state outputs need initial values.
- Intermediate outputs may omit initial values.
- Mutable defaults should use callables.
- `initial_state` is a runtime override, not a declaration change.
