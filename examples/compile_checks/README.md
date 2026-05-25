# Compile Checks

This directory contains small examples whose primary output is the compile
report. Some scripts are intentionally invalid; they demonstrate diagnostics
that are easier to inspect in isolation than inside a large model.

Regelum is built for feedback systems where each step commits state that shapes
later behavior. The paper motivation is that flexible PRS declarations make a
verification pass necessary before runtime: a model author can create mutually
dependent nodes in one phase, overlapping or missing transition guards, or a
phase graph that may never reach termination. Regelum checks these properties
at compile time before simulation starts.

Documentation: [what Regelum checks](../../docs/concepts/compilation.md#what-regelum-checks)
and [symbolic phase guards](../../docs/concepts/phases.md#branch-chains).

## What It Shows

| Script | Check | What it demonstrates |
| --- | --- | --- |
| `c1_violation.py` | C1 | Two nodes in one phase read each other's current state, so the phase dependency graph is cyclic. |
| `c3_violation.py` | C3 | Two identical guards overlap, so the next phase is ambiguous. |
| `c3_c2star_checks.py` | C3 and C2* | Prints an overlapping-guard diagnostic, then a cycle that passes bounded C2* reasoning. |
| `c2star_cycle.py` | C2* | Compares a dead symbolic cycle with a live cycle that must be rejected. |
| `c2star_split_writes.py` | C2* | Shows a two-phase cycle that is semantically dead after one traversal. |
| `unconditional_cycle.py` | C2 and C2* | Shows an unconditional phase cycle that is a definite infinite loop. |
| `complex_c3_partition.py` | C3 | Compares an exhaustive disjoint diagnostic partition with an overlapping one. |
| `complex_safety_loop.py` | C1 | Shows a valid sense/decide/apply/alarm loop and an invalid coupled control phase. |
| `bounded_loop.py` | Termination by topology | Unrolls a bounded loop into a finite acyclic phase chain. |

## What It Displays

Each script prints either `compile ok = True` for the accepted system or the
specific compile diagnostic for the rejected system. The diagnostics name the
phase and the violated structural condition.

Run one script:

```bash
uv run python examples/compile_checks/c1_violation.py
```

Run all compile-check tests:

```bash
uv run pytest tests/examples/compile_checks
```
