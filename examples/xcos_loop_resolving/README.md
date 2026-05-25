# Xcos Algebraic-Loop Resolution

This example reproduces a Scilab/Xcos-style direct-feedthrough feedback loop.
The literal block diagram has a sum block and gain block that need each
other's current output in the same instantaneous evaluation step. Regelum
rejects the literal one-phase PRS as a C1 phase dependency cycle, then shows
the phase-scheduled repair.

Regelum is built for feedback systems where each step commits state that shapes
later behavior. That pattern appears in closed-loop control, robotics, power
systems, autonomous driving, and industrial automation. Before simulation,
Regelum compiles the declarative PRS and checks that every phase has an
acyclic node dependency graph. This example is the direct analogue of an
algebraic-loop diagnostic in block-diagram tools.

Documentation: [what Regelum checks](../../docs/concepts/compilation.md#what-regelum-checks)
and [phase declarations](../../docs/concepts/phases.md).

## What It Shows

- `scilab_algebraic_loop.png` shows the literal Xcos diagram that cannot be
  ordered without a delay.
- `scilab_resolved.png` shows the conventional Xcos repair with an explicit
  delay-like block.
- `rg_reproduce_scilab.py` shows the Regelum equivalent: one invalid
  `literal-loop` phase and one valid `sum -> gain -> terminate` schedule.

## What It Displays

The script first prints the C1 diagnostic for the literal loop. It then prints
`compile ok = True` for the phase-scheduled model and five ticks of `sum`,
`gain`, and `trash` values.

Run it:

```bash
uv run python examples/xcos_loop_resolving/rg_reproduce_scilab.py
```
