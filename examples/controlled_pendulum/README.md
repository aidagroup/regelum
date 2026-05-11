# Controlled Pendulum

This example adds a swing-up controller to the pendulum plant. The plant is
integrated with `dt=0.01`; the controller runs with `dt=0.05`, so its torque
state is sampled and held between controller updates.

Run it as an interactive notebook:

```bash
uv run marimo edit examples/controlled_pendulum/rg-examples-controlled-pendulum.py
```

Run the same example as standalone Python:

```bash
uv run python examples/controlled_pendulum/standalone.py
```
