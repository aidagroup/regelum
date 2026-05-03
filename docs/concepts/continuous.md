# Continuous Dynamics

`regelum` can run ordinary discrete nodes and continuous ODE nodes in the same
`rg.PhasedReactiveSystem`.
The public system class does not change: a phase is treated as continuous when
all objects in `Phase.nodes` are `rg.ODESystem` instances.

## Continuous Nodes

Define continuous state with `rg.ODENode`.
An ODE node declares a `State` namespace instead of ordinary `Outputs`.
State variables are created with `rg.StateVar(initial=...)`, are readable by
other nodes, and are committed back into system state after integration.

```python
import casadi as ca
import regelum as rg


class Integrator(rg.ODENode):
    class Inputs(rg.NodeInputs):
        u: float = rg.Input(source=Controller.Outputs.u)

    class State(rg.NodeState):
        x: float = rg.StateVar(initial=0.0)

    def dstate(self, inputs: Inputs, state: State, *, time: object) -> State:
        return self.State(x=inputs.u + ca.sin(time))
```

`dstate(...)` returns the derivative in the same `State` shape.
It may be declared as either `dstate(self, inputs, state)` or
`dstate(self, inputs, state, *, time)`.
The `time` argument is the continuous solver time, not a sampled node output.

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
        tick: int = rg.Input(source=rg.Clock.tick)
        time: float = rg.Input(source=rg.Clock.time)
        x: float = rg.Input(source=Integrator.State.x)
```

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
outputs remain in state.

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
        time: float = rg.Input(source=rg.Clock.time)


rg.If(rg.V(rg.Clock.tick) >= 100, rg.terminate)
```

`snapshot()` returns user node outputs and ODE state values.
Use `read(rg.Clock.tick)` or `read(rg.Clock.time)` to inspect clock fields.
