# Connections

A connection tells an input which output it reads.
Connections may be declared on the input itself or attached to bound ports.

## Descriptor sources

The usual form is an input source declared inside the node class.
Most of the video player's wiring uses this form:

```python
class Decoder(rg.Node):
    class Inputs(rg.NodeInputs):
        bandwidth_kbps: float = rg.Input(source=Network.Outputs.bandwidth_kbps)
        bitrate_kbps: int = rg.Input(
            source=lambda: BitrateController.Outputs.value
        )
```

`Network` is already defined above, so a direct class reference works.
`BitrateController` is forward-referenced through a lazy `lambda`.

## Instance connections

Use instance connections when identity matters — multiple instances of the
same class in one phase, or wiring established outside the class body.

```python
session_main = MediaSession(name="main")
session_pip = MediaSession(name="pip")
policy = QualityPolicy()

rg.port(policy.Inputs.buffer_seconds).connect(session_main.Outputs.buffer_seconds)
```

The `port(...)` wrapper exposes `.connect(...)` while keeping the normal
descriptor syntax available for type checkers and readers.

## Direction

Connections must link an input to an output.
Both directions of `.connect(...)` are accepted; input-to-input and
output-to-output are errors.

```python
rg.port(policy.Inputs.buffer_seconds).connect(session_main.Outputs.buffer_seconds)
rg.port(session_main.Outputs.buffer_seconds).connect(policy.Inputs.buffer_seconds)
```

## Ambiguous class references

Class-level references become ambiguous when several producers match.
If the player ever holds two `MediaSession` instances, the existing
`source=lambda: MediaSession.Outputs.buffer_seconds` declaration in
`QualityPolicy` becomes ambiguous and compilation reports it.
The fix is an instance connection at construction time.

## Lazy and string sources

Lazy callables are useful for forward references and self-referential reads.
Self-referential reads are how the player gives `Clock`, `Network`, and
`BitrateController` their persistent state — the run method reads the previous
value and writes the next.

```python
class Clock(rg.Node):
    def run(
        self,
        tick: int = rg.Input(source=lambda: Clock.Outputs.tick),
    ) -> Outputs:
        return self.Outputs(tick=tick + 1)
```

String sources are an escape hatch.
Prefer real references when possible.
They survive renames better and give better compile diagnostics.

## Rules

- Prefer direct output references when the producer class is already known.
- Prefer lazy callables for forward references.
- Use `port(...).connect(...)` for multi-instance systems.
- Use instance-bound references to remove ambiguity.
