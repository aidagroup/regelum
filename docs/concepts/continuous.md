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
import numpy as np
import regelum as rg


class Integrator(rg.ODENode):
    class Inputs(rg.NodeInputs):
        u: np.ndarray = rg.Input(src=Controller.State.u)

    class State(rg.NodeState):
        x: np.ndarray = rg.Var(init=lambda: np.zeros(3))

    def dstate(self, inputs: Inputs, state: State, *, time: object) -> State:
        return self.State(x=A @ state.x + inputs.u + ca.sin(time))
```

`dstate(...)` returns the derivative in the same `State` shape.
`Var(init=...)` is the shape contract for continuous state. Scalars, 1D
vectors, and 2D matrices are supported. The runtime accepts `float`,
`list`, `tuple`, and `numpy.ndarray` values, but ODE state is always integrated
as floating point data.
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
The `time` value is continuous physical time inside the solver interval. It is
not the same thing as an input sourced from `rg.Clock.time`: `Input(src=Clock.time)`
is sampled once at the beginning of the ODE step, while `dstate(..., time)` varies
continuously during integration.

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

The ODE graph backend is CasADi. Regelum traces `dstate(...)` with `ca.MX`
values, including vector and matrix state. Use CasADi primitives directly inside
`dstate`, for example `ca.sin`, `ca.cos`, `ca.sqrt`, and `ca.if_else`.
Python `if`, `math`, and NumPy functions over symbolic state or input values are
not traceable. Plain vector algebra such as `A @ state.x`, `state.x + inputs.u`,
and scalar multiplication is supported when operands are CasADi-compatible.

## Vector State

Use `np.ndarray` when the model is naturally vector-valued:

```python
class Filter(rg.ODENode):
    class Inputs(rg.NodeInputs):
        voltage: np.ndarray = rg.Input(src=Inverter.State.phase_v)

    class State(rg.NodeState):
        current: np.ndarray = rg.Var(init=lambda: np.zeros(3))

    def dstate(self, inputs: Inputs, state: State) -> State:
        return self.State(current=(inputs.voltage - R * state.current) / L)
```

For CasADi tracing, `init=np.zeros(3)` becomes `MX(3, 1)` inside `dstate`.
`init=np.zeros((2, 3))` becomes `MX(2, 3)`.
After integration, Regelum converts the result back to the runtime type implied
by `init`, so `np.ndarray` state remains `np.ndarray` in `snapshot()` and
`read(...)`.

`list` and `tuple` state are also accepted. They are interpreted as CasADi
vectors/matrices during tracing and converted back to the same top-level
container after integration. Empty arrays, ragged nested lists, and rank greater
than 2 are rejected.

Input shapes are fixed when the CasADi graph is first traced. If an ODE input
later changes shape, Regelum raises an error instead of rebuilding the graph
silently.

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

`backend` selects the numerical integrator:

```python
electrical = rg.ODESystem(
    nodes=(plant,),
    dt="0.001",
    backend="casadi",
    method="cvodes",
    options={"abstol": 1e-9, "reltol": 1e-8},
)
```

The default is still `backend="scipy", method="LSODA"` for compatibility.
For `backend="scipy"`, `options` are passed to `scipy.integrate.solve_ivp`.
For `backend="casadi"`, `options` are passed to `casadi.integrator`.
`backend="casadi"` requires a CasADi integrator plugin such as `"cvodes"` or
`"rk"`; `"LSODA"` is a SciPy method and is rejected for the CasADi backend.

Regelum owns the integration interval. Do not pass CasADi time options
`t0`, `tf`, `grid`, or `output_t0`; they are rejected because the interval comes
from the PRS clock.

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
3. it integrates each `ODESystem` from the current clock time to
   `time + base_dt`;
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

If a system contains a continuous phase, the compiler also checks that the
phase route reaches a continuous phase exactly once per tick. This is a
symbolic feasibility check over transition guards: `update` implementations
are treated as arbitrary Python, so after each phase every state variable
written by that phase is modeled as a fresh symbolic value. A conditional route
that can terminate without the continuous phase, or can reach it twice, is a
compile error. Integrate continuous dynamics every base tick; use ordinary
`Node.dt` for slower discrete controllers whose state is sampled and held by
the plant.

With the CasADi numeric backend, Regelum caches `casadi.integrator` functions by
step duration. The integrator itself runs over local time `tau = 0..duration`;
the `time` argument passed to `dstate` is `time_start + tau`, so user equations
still see absolute physical time from the global clock.

## Base Time And Scheduling

Every system has a base time step.
The default is `base_dt="auto"`.

For a fully discrete system, `auto` means `base_dt = 1`.
If all explicitly scheduled discrete nodes have a larger common period, the
compiler emits an idle-tick warning and suggests an explicit `base_dt`.

For a system with continuous dynamics, `auto` computes the greatest common
divisor of all explicit discrete node `dt` values and all `ODESystem.dt`
values. `ODESystem.dt` contributes to the common time grid; runtime integration
still happens every tick over `base_dt`.
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

Continuous systems are not skipped by their own `dt`.
Each `ODESystem` in the continuous phase integrates every tick on `base_dt`.
This models sample-and-hold: slower discrete nodes update only on their
schedule, and continuous dynamics read their held state values between those
updates.

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
