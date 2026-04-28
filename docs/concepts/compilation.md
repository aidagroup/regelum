# Compilation

Compilation turns declarations into an executable system model.
It resolves references, checks consistency, and produces a compile report.

## What compilation resolves

Compilation resolves:

- node names;
- input sources;
- instance connections;
- output paths;
- phase targets;
- guard references;
- phase schedules;
- dependency edges;
- required initial outputs.

For the video player, the report's `phase_schedules` shows the topologically
ordered nodes per phase and `minimal_initial_outputs` lists the five outputs
that need to persist across ticks
(`Clock.tick`, `Network.bandwidth_kbps`, `BitrateController.value`,
`QualityPolicy.stalling`, `MediaSession.buffer_seconds`,
`Logger.history`).

## Compile report

Every system stores a `compile_report`.

```python
system = build_system()
report = system.compile_report

print(report.ok)
print(report.issues)
print(report.warnings)
print(report.phase_schedules)
print(report.minimal_initial_outputs)
```

Use `format()` for a compact text report.

```python
print(system.compile_report.format())
```

## Strict and non-strict systems

By default, invalid systems raise `CompileError`.

```python
system = rg.PhasedReactiveSystem(phases=phases)
```

Use `strict=False` to inspect invalid systems without raising.

```python
system = rg.PhasedReactiveSystem(
    phases=phases,
    strict=False,
)

print(system.compile_report.issues)
```

## Common compile issues

Typical issues include:

- input source is not connected;
- input source is unknown;
- class-level reference is ambiguous;
- output path is duplicated;
- explicit node names are duplicated;
- output without initial value is read too early;
- phase graph is incomplete;
- transition target is unknown;
- transition chain is malformed.

## C1 and C3 checks

Compilation rejects cyclic dependency graphs inside a phase.
This is the C1 check.
For the video player, the only non-trivial intra-phase dependency chain is
`Decoder → MediaSession → Logger` in `play`, which is acyclic.

For finite output domains, compilation also checks C3 by requiring exactly one
enabled transition per sampled state.
The branching in `decide` is `If(V(QualityPolicy.Outputs.stalling),
"drop_quality")` plus `Else("play")`, with `stalling: bool` — boolean has a
finite domain, so C3 is verified statically.

## Rules

- Read the compile report before debugging runtime behavior.
- Use `strict=False` for diagnostics.
- Resolve ambiguous class references with instance connections.
- Add initial values only for outputs that must exist before execution.
