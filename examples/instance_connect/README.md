# Instance Connections

This example demonstrates explicit post-instantiation connections with
`rg.port(...).connect(...)`. It creates two sources and two accumulators from
the same node classes, then connects each accumulator to a different source and
to its own previous total.

Regelum is built for feedback systems where each step commits state that shapes
later behavior. That pattern appears in closed-loop control, robotics, power
systems, autonomous driving, and industrial automation. Before simulation,
Regelum compiles the declarative PRS and checks that node identities, input
sources, phase schedules, and required initial state are unambiguous. This
example focuses on the case where class-level references would be ambiguous
because there are multiple instances of the same class.

Documentation: [post-instantiation connections](../../docs/concepts/nodes.md#post-instantiation-connections)
and [what compilation resolves](../../docs/concepts/compilation.md#what-compilation-resolves).

## What It Shows

- Two `NumberSource` instances publish independent values.
- Two `Accumulator` instances use the same class but keep separate runtime
  identities.
- Inputs are connected after object construction, so each accumulator reads the
  intended source and its own `total` state.

## What It Displays

The script prints the resolved input map from the compile report, the initial
snapshot, and the snapshot after three ticks. The final totals are `9` and `21`,
showing that the two accumulator instances remained independent.

Run it:

```bash
uv run python examples/instance_connect/instance_connect.py
```
