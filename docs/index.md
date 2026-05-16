---
hide:
  - toc
---

<style>
.md-content .md-typeset h1 { display: none; }
</style>

<p class="hero-logo">
  <img src="assets/logo/logo_big.png" alt="regelum">
</p>

<p>
  <a href="https://github.com/aidagroup/regelum">
    <img alt="repo" src="https://img.shields.io/badge/github-aidagroup%2Fregelum-181717?logo=github&logoColor=white">
  </a>
  <img alt="docs" src="https://img.shields.io/badge/docs-mkdocs%20material-526CFE?logo=materialformkdocs&logoColor=white">
  <img alt="tests" src="https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white">
  <img alt="uv" src="https://img.shields.io/badge/managed%20with-uv-6A5ACD">
</p>

`regelum` is a framework for designing and simulating dynamic systems, feedback
loops, DAGs, and dataflows as **Phased Reactive Systems** (PRS). A PRS has two
levels of abstraction: low-level computational primitives called **nodes**, and
high-level execution stages called **phases**. Nodes declare `Inputs` for values
read from other nodes or system sources and `State` for values they own and
publish; phases group node instances into executable slices, resolve their
dependency graph, and move execution through explicit transitions. These
transitions can be unconditional or conditional, so execution can branch to
different phases depending on current system state.

The API is intentionally built from Python's own language features: type
annotations, nested classes, descriptors, declarative function signatures,
operator overloading, and ordinary Python objects. It is inspired by frameworks
such as [FastAPI](https://fastapi.tiangolo.com/),
[Typer](https://typer.tiangolo.com/), [Pydantic](https://docs.pydantic.dev/),
[SQLModel](https://sqlmodel.tiangolo.com/), [SQLAlchemy](https://www.sqlalchemy.org/),
and [FastStream](https://faststream.ag2.ai/latest/). The goal is to make system
models look like regular Python while still giving the compiler enough
structure to resolve dependencies, validate the graph, schedule execution, and
integrate continuous dynamics.

## Overview

- **Nodes** declare typed inputs and state variables, then compute their next
  state in `update`.
- **Phases** decide which node instances are active together and how control
  moves between phases.
- **Continuous nodes** declare ODE state and are integrated through
  `ODESystem` phases.
- **Compilation** resolves links, schedules nodes, and catches structural
  mistakes before runtime: unresolved inputs, ambiguous references, invalid
  phase graphs, and computations that cannot be guaranteed to resolve.

Install from PyPI with `uv`:

```bash
uv add regelum
```

## Quick Example

```python
import regelum as rg


class TemperatureSensor(rg.Node):
    class State(rg.NodeState):
        temperature: float = rg.Var(init=19.0)

    def update(self) -> State:
        return self.State(temperature=21.5)


class HeaterController(rg.Node):
    class Inputs(rg.NodeInputs):
        temperature: float = rg.Input(
            src=TemperatureSensor.State.temperature,
        )

    class State(rg.NodeState):
        heater_on: bool

    def update(self, inputs: Inputs) -> State:
        return self.State(heater_on=inputs.temperature < 22.0)


class HeatingCycles(rg.Node):
    class Inputs(rg.NodeInputs):
        heater_on: bool = rg.Input(src=HeaterController.State.heater_on)

    class State(rg.NodeState):
        count: int = rg.Var(init=0)

    def update(self, inputs: Inputs, prev_state: State) -> State:
        return self.State(
            count=prev_state.count + int(inputs.heater_on),
        )


sensor = TemperatureSensor(name="room_sensor")
controller = HeaterController(name="heater_controller")
cycles = HeatingCycles(name="heating_cycles")

system = rg.PhasedReactiveSystem(
    phases=[
        rg.Phase(
            "control",
            nodes=(sensor, controller, cycles),
            transitions=(rg.Goto(rg.terminate),),
            is_initial=True,
        ),
    ],
)

system.step()
print(system.read(controller.State.heater_on))
```

## Local development

Set up the repository, install hooks, and run the full local gate:

```bash
uv sync --all-groups
uv run prek install --hook-type pre-commit --hook-type pre-push
uv run prek run --all-files
```

The `pre-push` hook runs `pytest`, so a normal `git push` exercises the test
suite automatically.

## Local docs

```bash
uv run --group docs mkdocs serve \
  --watch docs \
  --watch mkdocs.yml \
  --watch src/regelum \
  --watch README.md
```

Build the static site:

```bash
uv run --group docs mkdocs build
```
