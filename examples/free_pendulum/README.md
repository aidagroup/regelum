# Free Pendulum

This example defines a damped pendulum plant and an observer in a small phased
reactive system. The plant is integrated with `dt=0.01`; the observer publishes
`sin(theta)`, `cos(theta)`, and angular velocity.

Run it as an interactive notebook:

```bash
uv run marimo edit examples/free_pendulum/app.py
```

Run the same example as standalone Python:

```bash
uv run python examples/free_pendulum/standalone.py
```
