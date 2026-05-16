# Two Inverter Static Droop Control

This example simulates a native two-inverter microgrid inspired by the
OpenModelica Microgrid Gym static droop control example. The master inverter is
voltage-forming, the slave inverter is current-sourcing, and both feed a common
three-phase bus and load through LC/LCL branches.

The standalone script uses NumPy arrays for three-phase values and the CasADi
ODE backend for the electrical plant. A discrete resistance scenario node drives
the load through nominal resistance, doubled resistance, and nominal resistance
again.

Run the standalone example:

```bash
uv run python examples/two_inverter_static_droop/standalone.py
```

Generate the documentation plot:

```bash
uv run python examples/two_inverter_static_droop/standalone.py \
  --steps 2000
```

The command writes `lcl1_voltage_and_resistance.svg` in this example directory
and copies the same plot into the documentation assets.
