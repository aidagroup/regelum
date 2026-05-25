# Two Inverter Static Droop Control

This example simulates a native two-inverter microgrid inspired by the
OpenModelica Microgrid Gym static droop control example. The master inverter is
voltage-forming, the slave inverter is current-sourcing, and both feed a common
three-phase bus and load through LC/LCL branches.

The standalone script uses NumPy arrays for three-phase values and the CasADi
ODE backend for the electrical plant. A discrete resistance scenario node drives
the load through nominal resistance, doubled resistance, and nominal resistance
again.

Regelum is built for feedback systems where each step commits state that shapes
later behavior. That pattern appears in closed-loop control, robotics, power
systems, autonomous driving, and industrial automation. Before simulation,
Regelum compiles the declarative PRS and checks that phases can be scheduled,
guards are deterministic where they are symbolic, and each tick reaches
termination. This example exercises those checks on a larger cyber-physical
model with many controller nodes reading and writing vector-valued electrical
state.

Documentation: [two-inverter walkthrough](../../docs/examples/two_inverter_static_droop.md)
and [continuous dynamics](../../docs/concepts/continuous.md).

## What It Shows

- Master inverter direct droop control for frequency and voltage setpoints.
- Slave inverter PLL, inverse droop, and current-control loop.
- One CasADi-backed ODE phase for the electrical network.
- A discrete load-resistance scenario that changes during the run.

## What It Displays

The script prints simulation status and writes
`lcl1_voltage_and_resistance.svg`, showing the LCL branch capacitor phase
voltages together with the switched load resistance.

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
