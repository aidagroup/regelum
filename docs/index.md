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

## Quick Example

Install from PyPI with `uv`:

```bash
uv add regelum
```

```python
import random

import regelum as rg


class TemperatureSensor(rg.Node):
    class State(rg.NodeState):
        temperature: float

    def update(self) -> State:
        return self.State(temperature=round(random.uniform(18.0, 26.0), 1))


class HeaterController(rg.Node):
    class Inputs(rg.NodeInputs):
        temperature: float = rg.src(TemperatureSensor.State.temperature)

    class State(rg.NodeState):
        heater_on: bool

    def update(self, inputs: Inputs) -> State:
        return self.State(heater_on=inputs.temperature < 22.0)


class HeatingCycles(rg.Node):
    class State(rg.NodeState):
        count: int = rg.var(init=0)

    def update(
        self,
        prev_state: State,
        is_heater_on: bool = rg.src(HeaterController.State.heater_on),
    ) -> State:
        return self.State(
            count=prev_state.count + int(is_heater_on),
        )


sensor = TemperatureSensor()
controller = HeaterController()
cycles = HeatingCycles()

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
