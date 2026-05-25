# Free Pendulum

This example defines a damped torque-free pendulum plant and an observer in a
small phased reactive system. The plant is integrated with `dt=0.01`; the
observer publishes `sin(theta)`, `cos(theta)`, and angular velocity.

Regelum is built for feedback systems where each step commits state that shapes
later behavior. That pattern appears in closed-loop control, robotics, power
systems, autonomous driving, and industrial automation. Before simulation,
Regelum compiles the declarative PRS and checks that phases can be scheduled,
guards are deterministic where they are symbolic, and each tick reaches
termination. This example is intentionally simple: it demonstrates a valid
continuous/discrete phase split.

Documentation: [free pendulum walkthrough](../../docs/examples/free_pendulum.md)
and [continuous dynamics](../../docs/concepts/continuous.md).

## What It Shows

- `FreePendulum` is an `ODENode` with physical state `theta` and `omega`.
- `Observer` reads the ODE state and publishes trigonometric features.
- `Logger` records time, angle features, and angular velocity.
- The phase graph runs `plant -> observe -> terminate`.

## What It Displays

The standalone script prints the final simulated time, angle, sine/cosine
features, and angular velocity after the default run.

Run it as an interactive notebook:

```bash
uv run marimo edit examples/free_pendulum/rg-examples-free-pendulum.py
```

Run the same example as standalone Python:

```bash
uv run python examples/free_pendulum/standalone.py
```
