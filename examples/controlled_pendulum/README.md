# Controlled Pendulum

This example stabilizes a pendulum with a clipped PD controller. The plant is
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

Save the response plot:

```bash
uv run python examples/controlled_pendulum/standalone.py --output artifacts/controlled_pendulum_response.png
```
