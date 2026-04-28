# Ports

Ports are typed declarations on node namespaces.
Inputs read values.
Outputs write values.

## Input namespaces

Inputs are declared by subclassing `rg.NodeInputs`.

```python
class Inputs(rg.NodeInputs):
    bandwidth_kbps: float = rg.Input(source=Network.Outputs.bandwidth_kbps)
    bitrate_kbps: int = rg.Input(
        source=lambda: BitrateController.Outputs.value
    )
```

The first declaration uses a direct class reference because `Network` is
already in scope.
The second uses a lazy callable to avoid a forward-reference cycle:
`BitrateController` reads its own previous value, and `Decoder` reads
`BitrateController` while still being defined above it.

A bare annotation is shorthand for an unconnected input.

```python
class Inputs(rg.NodeInputs):
    value: float
```

This is equivalent to:

```python
class Inputs(rg.NodeInputs):
    value: float = rg.Input()
```

Unconnected inputs are compile errors unless connected later.

## Output namespaces

Outputs are declared by subclassing `rg.NodeOutputs`.

```python
class Outputs(rg.NodeOutputs):
    bandwidth_kbps: float = rg.Output(initial=2160.0)
```

A bare annotation is shorthand for an output without an initial value:

```python
class Outputs(rg.NodeOutputs):
    fetched_seconds: float
```

`Decoder.fetched_seconds` is declared this way: it is produced freshly on
every tick by `Decoder` before `MediaSession` reads it in the same phase, so
no initial value is needed.
Compilation rejects bare outputs that are read before they can be produced.

## Namespace names

The conventional names are `Inputs` and `Outputs`.
The runtime detects namespaces by base class, not by name.

```python
class Source(rg.Node):
    class Vars(rg.NodeOutputs):
        value: int = rg.Output(initial=0)
```

There is no canonical alias.
If a node declares `Vars`, references must use `Vars`.

Each node may declare at most one input namespace and at most one output
namespace.

## Class references and instance references

`Network.Outputs.bandwidth_kbps` is a class-level output reference.
It is concise and works when only one `Network` instance is present.

`network.Outputs.bandwidth_kbps` is an instance-bound output reference.
It identifies a concrete producer instance.

Use instance-bound references when a phase contains multiple instances of the
same node class — e.g., a multi-stream player with two `MediaSession`
instances would need `session_main.Outputs.buffer_seconds` and
`session_pip.Outputs.buffer_seconds` to disambiguate.

## Lazy references

Use a zero-argument callable for forward references.
The video player uses this pattern wherever a node reads either its own
previous value or a peer that is defined later in the file:

```python
class Inputs(rg.NodeInputs):
    bitrate_kbps: int = rg.Input(
        source=lambda: BitrateController.Outputs.value
    )
```

Use a one-argument callable for instance-dependent initial values.
The argument name is arbitrary; `self` is the recommended spelling.

```python
buffer_seconds: float = rg.Output(
    initial=lambda self: cast(MediaSession, self).initial_buffer,
)
```

## Rules

- Inputs read outputs.
- Outputs define state values.
- Bare input annotations create unconnected inputs.
- Bare output annotations create outputs without initial values.
- Class-level references must resolve to exactly one producer.
- Instance-bound references avoid ambiguity.
