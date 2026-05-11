# Two Inverter Static Droop Control

This example simulates a native two-inverter microgrid inspired by the
OpenModelica Microgrid Gym static droop control example. The master inverter is
voltage-forming, the slave inverter is current-sourcing, and both feed a common
three-phase bus and load through LC/LCL branches.

Run the standalone example:

```bash
uv run python examples/two_inverter_static_droop/standalone.py
```

Generate the documentation plot:

```bash
uv run python examples/two_inverter_static_droop/standalone.py \
  --steps 1000 \
  --output docs/assets/examples/two_inverter_static_droop/lcl1_capacitor_voltages.svg
```
