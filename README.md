# regelum

Minimal prototype for a node-based runtime with typed input and output namespaces.

The walkthrough example used by the docs is an adaptive-bitrate video player —
a closed-loop system where each tick decides whether to keep playing or to
drop quality first based on buffer level and bandwidth:

```bash
uv run regelum-video-player
```

See `docs/concepts/` for a step-by-step introduction (nodes → ports → state →
phases → transitions → predicates → tick lifecycle) anchored to this example.

Run the pendulum example:

```bash
uv run regelum-pendulum
```

Run the JSON enrichment example:

```bash
uv run regelum-json-enrichment
```

Run the instance identity and `connect(...)` example:

```bash
uv run regelum-instance-connect
```

Run the C1 violation example:

```bash
uv run python -m examples.c1_violation
```

Run the C3 violation example:

```bash
uv run python -m examples.c3_violation
```

Run the complex ok/bad example:

```bash
uv run python -m examples.complex_safety_loop
```

Run the complex C3 partition example:

```bash
uv run python -m examples.complex_c3_partition
```

Run tests:

```bash
uv run pytest
```

Run type checking:

```bash
uv run pyright src tests/typecheck_api.py
```

`PhasedReactiveSystem(...)` is defined by phases.
The node set is inferred from the node instances assigned to `Phase.nodes`.
`Phase.nodes` accepts node instances only, not node classes.
Compilation fails on unlinked inputs, unknown sources, duplicate outputs, and
outputs that are read before they can be produced without an initial value.
The resulting `compile_report` exposes resolved inputs, phase schedules,
dependency edges, outputs without initial values, and required initial outputs.
It also checks C1 by rejecting cyclic dependency graphs inside each phase.
It checks C3 for finite output domains, including boolean outputs, by requiring exactly one enabled transition per sampled state.

Use `strict=False` to inspect an invalid system without raising `CompileError`:

```python
system = PhasedReactiveSystem(phases=phases, strict=False)
report = system.compile_report

print(report.ok)
print(report.issues)
print(report.warnings)
print(report.minimal_initial_outputs)
print(report.required_initial_outputs)
print(report.format())
```

Override tick-0 values with `initial_state`.
The override supplies values for one run without mutating node configuration:

```python
plant = Plant()

system = PhasedReactiveSystem(
    phases=[
        Phase("tick", nodes=(plant,), transitions=(Goto(terminate),), is_initial=True),
    ],
    initial_state={
        Plant.Outputs.theta: 1.57,
        Plant.Outputs.omega: 0.0,
    },
)

system.reset(
    initial_state={
        Plant.Outputs.theta: 0.35,
    }
)
```

Typed port declarations use field-style descriptors.
Use output objects when the producing class is already available.
Use zero-argument callables for forward, self, or cyclic dependencies so IDE rename refactors still see real attribute references.
String references remain available as an escape hatch.
Bare annotations are shorthand for default ports:

```python
class Inputs(NodeInputs):
    unconnected_value: float
    theta: float = Input(source=PendulumPlant.Outputs.theta)
    torque: float = Input(source=lambda: PDController.Outputs.torque)
    previous_theta: float = Input(source=lambda: PendulumPlant.Outputs.theta)

class Outputs(NodeOutputs):
    torque: float
    theta: float = Output(initial=0.35)
```

`value: T` inside `Inputs` is equivalent to `value: T = Input()`.
`value: T` inside `Outputs` is equivalent to `value: T = Output()`.
For compact nodes, inputs may also be declared directly on `run(...)`.
Do not mix this style with `class Inputs` in the same node.

```python
class Sink(Node):
    class Outputs(NodeOutputs):
        seen: int

    def run(
        self,
        value: int = Input(source=lambda: Source.Outputs.value),
    ) -> Outputs:
        return self.Outputs(seen=value)
```

The conventional namespace names are `Inputs` and `Outputs`, but the runtime
detects namespaces by their base classes, not by their names.
No canonical alias is created: if a node declares `Vars`, use `Vars`.

```python
class Source(Node):
    class Vars(NodeOutputs):
        value: int = Output(initial=1)

    def run(self) -> Vars:
        return self.Vars(value=2)

class Sink(Node):
    class In(NodeInputs):
        value: int = Input(source=lambda: Source.Vars.value)

    class Out(NodeOutputs):
        seen: int

    def run(self, inputs: In) -> Out:
        return self.Out(seen=inputs.value)
```

Each node may declare zero or one `NodeInputs` namespace and zero or one
`NodeOutputs` namespace.
Subclasses inherit a namespace when they do not declare their own; declaring a
new namespace overrides the inherited one.
String output references may use either `Source.value` or the explicit
namespace form such as `Source.Outputs.value` / `Source.Vars.value`; all forms
normalize to the same internal path.

For multiple instances of the same node class, give instances stable names and
connect instance-bound ports explicitly:

```python
source_a = Source(name="source_a")
source_b = Source(name="source_b")
sink_a = Sink(name="sink_a")
sink_b = Sink(name="sink_b")

port(sink_a.Inputs.value).connect(source_a.Outputs.value)
port(source_b.Outputs.value).connect(sink_b.Inputs.value)

system = PhasedReactiveSystem(
    [source_a, source_b, sink_a, sink_b],
    phases=[
        Phase(
            "copy",
            nodes=(source_a, source_b, sink_a, sink_b),
            transitions=(Goto(terminate),),
            is_initial=True,
        ),
    ],
)
```

Initial state belongs to output declarations.
Static initial values are direct:

```python
theta: float = Output(initial=0.35)
```

Outputs that are produced before their first read in the same phase do not need
an initial value:

```python
class Source(Node):
    class Outputs(NodeOutputs):
        value: int

    def run(self) -> Outputs:
        return self.Outputs(value=5)

class Sink(Node):
    class Inputs(NodeInputs):
        value: int = Input(source=lambda: Source.Outputs.value)
```

Outputs used as previous state still need an initial value:

```python
previous_theta: float = Input(source=lambda: Plant.Outputs.theta)
theta: float = Output(initial=0.35)
```

Initial values can also be computed by zero-argument callables:

```python
theta: float = Output(initial=lambda: 0.35)
```

For values that belong to a concrete node instance, use a one-argument callable.
The argument name is arbitrary; `self` is the recommended spelling:

```python
from typing import cast

from regelum import Node, NodeOutputs, Output

class Plant(Node):
    def __init__(
        self,
        init_theta: float = 0.35,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.init_theta = init_theta

    class Outputs(NodeOutputs):
        theta: float = Output(initial=lambda self: cast(Plant, self).init_theta)
```

Class-level initial values are suitable only for shared defaults.
Instance-dependent initial values should use `lambda self: cast(NodeType, self)...`.

Runtime state stores Python objects by reference.
Use callable initial values for fresh containers and pass the same object
through an enrichment pipeline when in-place updates are intentional:

```python
JsonDoc = dict[str, object]

class Source(Node):
    class Outputs(NodeOutputs):
        doc: JsonDoc = Output(initial=lambda: {})

    def run(self) -> Outputs:
        return self.Outputs(doc={"raw": {"text": "hello"}})

class Enricher(Node):
    class Inputs(NodeInputs):
        doc: JsonDoc = Input(source=lambda: Source.Outputs.doc)

    class Outputs(NodeOutputs):
        doc: JsonDoc = Output(initial=lambda: {})

    def run(self, inputs: Inputs) -> Outputs:
        inputs.doc["features"] = {"length": 5}
        return self.Outputs(doc=inputs.doc)
```

Ticks are phase driven.
Each phase runs its active nodes, then exactly one transition predicate must be true:
Within a phase, nodes are executed in topological order inferred from input
dependencies; the order in `Phase.nodes` is only used to break ties between
independent nodes.

```python
controller = PDController()
plant = PendulumPlant()
logger = Logger()

system = PhasedReactiveSystem(
    phases=[
        Phase(
            "control",
            nodes=(controller, logger),
            transitions=(Goto("plant"),),
            is_initial=True,
        ),
        Phase("plant", nodes=(plant,), transitions=(Goto(terminate),)),
    ],
)
```

Predicates inspect the committed state snapshot:

```python
Phase(
    "source",
    nodes=(source,),
    transitions=(
        If(V(Source.Outputs.value) > 1.0, "sink", name="value-ready"),
        Else(terminate),
    ),
)
```

Use `Goto(target)` for an unconditional transition.
Use `If(...)`, any number of `ElseIf(...)`/`Elif(...)`, and an optional final `Else(...)`
for an ordered branch chain.
The special target `terminate` ends the current tick.

`V(...)` accepts the same output references as inputs: output descriptors,
instance-bound output ports, lazy callables, and string references.
Enum outputs are supported directly when the output annotation is an enum class:

```python
from enum import Enum


class ModeValue(Enum):
    IDLE = "idle"
    ACTIVE = "active"


class Mode(Node):
    class Outputs(NodeOutputs):
        value: ModeValue = Output(initial=ModeValue.IDLE)


If(V(Mode.Outputs.value) == ModeValue.ACTIVE, "active", name="active")
```

Use `Goto(terminate)` for an unconditional transition to tick completion.
