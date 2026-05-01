<p align="center">
  <img
    src="https://raw.githubusercontent.com/aidagroup/regelum/main/logo/logo_big.png"
    alt="regelum"
    width="360"
  >
</p>

<p align="center">
  <a href="https://aidagroup.github.io/regelum/">
    <img alt="docs" src="https://img.shields.io/badge/docs-mkdocs%20material-526CFE?logo=materialformkdocs&logoColor=white">
  </a>
  <img alt="tests" src="https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white">
  <img alt="python" src="https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white">
  <img alt="uv" src="https://img.shields.io/badge/managed%20with-uv-6A5ACD">
</p>

# regelum

`regelum` is a framework for modeling **Phased Reactive Systems**:
systems that execute one tick at a time, activate different groups of nodes in
different phases, and move between phases with explicit transitions.

## Overview

- **Nodes** declare typed inputs and outputs.
- **Phases** decide which node instances are active together.
- **Transitions** describe how one phase hands control to the next.
- **Compilation** resolves links, schedules nodes, and catches structural
  mistakes before runtime.

The best entry point is the concepts overview:

- Docs: <https://aidagroup.github.io/regelum/>
- Concepts overview: <https://aidagroup.github.io/regelum/concepts/>

## Quick Example

```python
import regelum as rg


class TemperatureSensor(rg.Node):
    class Outputs(rg.NodeOutputs):
        temperature: float = rg.Output(initial=19.0)

    def run(self) -> Outputs:
        return self.Outputs(temperature=21.5)


class HeaterController(rg.Node):
    class Inputs(rg.NodeInputs):
        temperature: float = rg.Input(
            source=TemperatureSensor.Outputs.temperature,
        )

    class Outputs(rg.NodeOutputs):
        heater_on: bool

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(heater_on=inputs.temperature < 22.0)


sensor = TemperatureSensor(name="room_sensor")
controller = HeaterController(name="heater_controller")

system = rg.PhasedReactiveSystem(
    phases=[
        rg.Phase(
            "control",
            nodes=(sensor, controller),
            transitions=(rg.Goto(rg.terminate),),
            is_initial=True,
        ),
    ],
)

snapshot = system.step()
print(snapshot["heater_controller.heater_on"])
```

## Installation

Recommended: use `uv`.

```bash
uv add regelum
```

For local development from this repository:

```bash
uv sync --all-groups
uv run pytest tests
uv run ty check src tests
uv run mkdocs serve
```

## Examples

```bash
uv run regelum-pendulum
uv run regelum-video-player
uv run regelum-instance-connect
```

## Release Process

Create a GitHub Release with a tag such as `v0.2.0`.

The publish workflow will:

1. build the package with `uv build`
2. derive version `0.2.0` from the git tag via `hatch-vcs`
3. publish the artifacts to PyPI with `uv publish`

After installation, users can verify the packaged version:

```python
import regelum

print(regelum.__version__)
```
