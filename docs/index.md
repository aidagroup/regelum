<style>
.md-content .md-typeset h1 { display: none; }
</style>

<p class="hero-logo">
  <img
    class="hero-logo-light"
    src="assets/logo_big_transparent.png"
    alt="regelum"
  >
  <img
    class="hero-logo-dark"
    src="assets/logo_big_white_transparent.png"
    alt="regelum"
  >
</p>

`regelum` is a small prototype runtime for typed node systems,
explicit data-flow links, and phase-based reactive execution.

The framework is designed around three surface ideas:

- nodes declare typed input and output ports;
- phases define which node instances are active together;
- transitions describe phase switching with symbolic guards.

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
```

## Local docs

```bash
uv run --group docs mkdocs serve \
  --watch docs \
  --watch mkdocs.yml \
  --watch src/regelum \
  --watch logo \
  --watch README.md
```

Build the static site:

```bash
uv run --group docs mkdocs build
```
