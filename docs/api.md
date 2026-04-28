# API Sketch

## Nodes

Nodes define input and output namespaces.
Bare annotations are accepted as default ports.

```python
class Controller(Node):
    class Inputs(NodeInputs):
        theta: float = Input(source=Plant.Outputs.theta)
        omega: float = Input(source=Plant.Outputs.omega)

    class Outputs(NodeOutputs):
        torque: float
```

Compact nodes may declare inputs directly on `run`.
Do not mix this style with a `NodeInputs` subclass in the same node.

```python
class Logger(Node):
    class Outputs(NodeOutputs):
        samples: int = Output(initial=0)

    def run(self, theta: float = Input(source=Plant.Outputs.theta)) -> Outputs:
        return self.Outputs(samples=self.samples + 1)
```

## Phases

`PhasedReactiveSystem` is defined by phases.
Each phase lists node instances, not node classes.

```python
system = PhasedReactiveSystem(
    phases=[
        Phase("control", nodes=(plant, controller), transitions=(Goto(terminate),), is_initial=True),
    ],
)
```

## Transitions

Use `If`, `Elif`, `Else`, and `Goto` for phase switching.

```python
transitions = (
    If(V(sensor.Outputs.ready), "run"),
    Elif(V(sensor.Outputs.failed), "fault"),
    Else(terminate),
)
```
