# Controlled Pendulum

This example stabilizes a pendulum with a clipped PD controller. The plant is
integrated with `dt=0.01`; the controller runs with `dt=0.05`, so its torque
state is sampled and held between controller updates.

Regelum is built for feedback systems where each step commits state that shapes
later behavior. That pattern appears in closed-loop control, robotics, power
systems, autonomous driving, and industrial automation. Before simulation,
Regelum compiles the declarative PRS and checks that phases can be scheduled,
guards are deterministic where they are symbolic, and each tick reaches
termination. In this example those checks cover the mixed continuous/discrete
schedule and the sampled controller state read by the plant.

Documentation: [controlled pendulum walkthrough](../../docs/examples/controlled_pendulum.md),
[phases](../../docs/concepts/phases.md), and
[continuous dynamics](../../docs/concepts/continuous.md).

## What It Shows

- `PendulumODE` integrates the physical angle and angular velocity.
- `Observer` converts raw angle state into `sin(theta)`, `cos(theta)`, and
  `omega`.
- `Controller` computes a saturated PD torque every 0.05 seconds.
- `Logger` records the closed-loop response for plotting.

## What It Displays

The standalone script prints the final time, angle, angular velocity, and held
torque. With `--output`, it writes a plot of angle, angular velocity, and
sample-and-hold torque.

Run it as an interactive notebook:

```bash
uv run marimo edit examples/controlled_pendulum/rg-examples-controlled-pendulum.py
```

Run the same example as standalone Python:

```bash
uv run python examples/controlled_pendulum/standalone.py
```

Save the response plot:

```bash
uv run python examples/controlled_pendulum/standalone.py --output artifacts/controlled_pendulum_response.png
```
