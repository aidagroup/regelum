# Continuous Dynamics

`regelum` can run ordinary discrete nodes and continuous ODE nodes in the same
`rg.PhasedReactiveSystem`.
The public system class does not change: a phase is treated as continuous when
all objects in `Phase.nodes` are `rg.ODESystem` instances.

## Continuous Nodes

Define continuous state with `rg.ODENode`.
An ODE node declares a `State` namespace instead of ordinary `State`.
State variables are created with `rg.Var(init=...)`, are readable by
other nodes, and are committed back into system state after integration.

```python
import casadi as ca
import regelum as rg


class Integrator(rg.ODENode):
    class Inputs(rg.NodeInputs):
        u: float = rg.Input(src=Controller.State.u)

    class State(rg.NodeState):
        x: float = rg.Var(init=0.0)

    def dstate(self, inputs: Inputs, state: State, *, time: object) -> State:
        return self.State(x=inputs.u + ca.sin(time))
```

`dstate(...)` returns the derivative in the same `State` shape.
Declare only the arguments the node actually needs.
The ODE runtime resolves `inputs` and `state` by name or by annotation.
`time` is a reserved name and is resolved by name.
Individual input ports can also be declared directly on `dstate` with
`rg.Input(...)`.
Arguments may be declared in any order, so these forms are supported:

```python
def dstate(self, inputs, state, time): ...
def dstate(self, time, state: State, inputs: Inputs): ...
def dstate(self, control: Inputs, memory: State): ...
def dstate(self, time, state: State, a=rg.Input(...), b=rg.Input(...)): ...
def dstate(self, inputs, state): ...
def dstate(self, inputs, time): ...
def dstate(self, time): ...
def dstate(self, inputs): ...
def dstate(self, state): ...
```

`time` may also be keyword-only, for example
`dstate(self, inputs, state, *, time)`.
The `time` value is the continuous solver time, not a sampled node state variable.

When `dstate` declares input ports directly, those parameter names become the
ODE node input names and can be connected like ordinary inputs:

```python
class Plant(rg.ODENode):
    class State(rg.NodeState):
        x: float = rg.Var(init=0.0)

    def dstate(
        self,
        time,
        state: State,
        u: float = rg.Input(src=Controller.State.u),
    ) -> State:
        return self.State(x=u - state.x + ca.sin(time))
```

Sources may be lazy references, just like ordinary node inputs:

```python
def dstate(
    self,
    state: State,
    load: float = rg.Input(src=lambda: Load.State.current),
) -> State:
    ...
```

The ODE backend is CasADi-only.
Use CasADi primitives directly inside `dstate`, for example `ca.sin`,
`ca.cos`, `ca.sqrt`, and `ca.if_else`.
Python `if`, `math`, and NumPy operations over symbolic state or input values
are not traceable by the backend.

## ODE Systems

Group one or more `ODENode` instances into an `rg.ODESystem`:

```python
plant = Integrator()
electrical = rg.ODESystem(
    nodes=(plant,),
    dt="0.001",
    method="LSODA",
)
```

`dt` is required for `ODESystem`.
Pass it as a `Fraction`, an integer, or a decimal string.
Floats are rejected so the runtime can compute exact scheduling ratios.
Do not put `dt` on `ODENode`.
An `ODENode` is an equation block inside an `ODESystem`, not a separately
scheduled node; both instance-level `Integrator(dt="0.001")` and class-level
`class Integrator(rg.ODENode): dt = "0.001"` are rejected.

An ODE phase contains only ODE systems:

```python
system = rg.PhasedReactiveSystem(
    phases=[
        rg.Phase("control", nodes=(controller,), transitions=(rg.Goto("plant"),), is_initial=True),
        rg.Phase("plant", nodes=(electrical,), transitions=(rg.Goto("log"),)),
        rg.Phase("log", nodes=(logger,), transitions=(rg.Goto(rg.terminate),)),
    ],
)
```

A phase may not mix `ODESystem` objects with ordinary `Node` objects.
For now, one `PhasedReactiveSystem` supports at most one continuous phase.
One continuous phase may contain more than one independent `ODESystem`, but
cross-system continuous coupling is rejected at compile time.
If an `ODENode` in one `ODESystem` reads state from an `ODENode` in another
`ODESystem`, put those ODE nodes into the same `ODESystem`.

## Resolution Order

One tick still walks the phase graph from the initial phase to `terminate`.
Discrete phases run the due ordinary nodes in compiled topological order.
When the runtime reaches a continuous phase:

1. it builds the current input snapshot;
2. it traces or reuses the CasADi graph `f(t, x, p)` and its Jacobian;
3. it integrates each due `ODESystem` from the current clock time to
   `time + ODESystem.dt`;
4. it writes every internal `ODENode.State` value into system state;
5. it advances `Clock.time` immediately, before the next phase runs.

The integer `Clock.tick` advances only after the full PRS tick terminates.
That means a logger after a continuous phase sees the new physical time and
the old tick index:

```python
class Logger(rg.Node):
    class Inputs(rg.NodeInputs):
        tick: int = rg.Input(src=rg.Clock.tick)
        time: float = rg.Input(src=rg.Clock.time)
        x: float = rg.Input(src=Integrator.State.x)
```

Multiple `ODESystem` objects in the same continuous phase are treated as
separate systems. They are valid only when they are independent. Continuous
state dependencies across `ODESystem` boundaries would imply operator
splitting/sample-and-hold semantics, so `regelum` currently rejects them
instead of silently choosing an execution order.

## Base Time And Scheduling

Every system has a base time step.
The default is `base_dt="auto"`.

For a fully discrete system, `auto` means `base_dt = 1`.
If all explicitly scheduled discrete nodes have a larger common period, the
compiler emits an idle-tick warning and suggests an explicit `base_dt`.

For a system with continuous dynamics, `auto` computes the greatest common
divisor of all explicit discrete node `dt` values and all `ODESystem.dt`
values.
Every node period must be an integer multiple of `base_dt`.

```python
controller = Controller(dt="0.01")
electrical = rg.ODESystem(nodes=(plant,), dt="0.001")

system = rg.PhasedReactiveSystem(
    phases=phases,
    base_dt="auto",  # Fraction(1, 1000)
)
```

Discrete nodes with `dt` use sample-and-hold semantics.
If a node is not due on the current base tick, it is skipped and its previous
state variables remain in state.

```python
fast = Sensor()          # due every base tick
slow = Controller(dt=2)  # due on ticks 0, 2, 4, ...
```

The same scheduling rule applies to continuous systems: `period_ticks` is
`dt / base_dt`, and due checks use `Clock.tick % period_ticks == 0`.

## System Clock

The system clock is a reserved source, not a node.
Use `rg.Clock.tick` and `rg.Clock.time` in inputs and guards:

```python
class Sampler(rg.Node):
    class Inputs(rg.NodeInputs):
        time: float = rg.Input(src=rg.Clock.time)


rg.If(rg.V(rg.Clock.tick) >= 100, rg.terminate)
```

`snapshot()` returns user node state variables and ODE state values.
Use `read(rg.Clock.tick)` or `read(rg.Clock.time)` to inspect clock fields.
