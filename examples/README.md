# Regelum Examples

These examples are local repository examples. They are not packaged into the
PyPI distribution; run them from a checkout of the repository.

Regelum targets feedback systems where the next step depends on state produced
by previous steps. The paper motivation is that this structure is common in
closed-loop control, robotics, power systems, autonomous driving, and industrial
automation. The same flexibility also creates verification obligations before
runtime: a model author can write mutually dependent nodes in one phase,
overlapping or incomplete guarded transitions, or a phase graph that never
reaches termination. Regelum therefore compiles and checks the PRS structure
before simulation.

At compile time, Regelum checks node and phase consistency, phase-local
schedulability, guarded-transition determinism, and tick termination or
cycle-resolution conditions. See the documentation for the corresponding
[node model](../docs/concepts/nodes.md), [phase model](../docs/concepts/phases.md),
and [compile report](../docs/concepts/compilation.md).

| Example | What it shows |
| --- | --- |
| [`free_pendulum`](free_pendulum/) | A torque-free continuous pendulum, observer, logger, and response printout. |
| [`controlled_pendulum`](controlled_pendulum/) | Closed-loop pendulum stabilization with sampled controller torque and a response plot. |
| [`two_inverter_static_droop`](two_inverter_static_droop/) | A larger three-phase microgrid simulation with static droop control and SVG output. |
| [`video_player`](video_player/) | A discrete adaptive-bitrate feedback loop with conditional phase branching. |
| [`instance_connect`](instance_connect/) | Post-instantiation port connections for multiple instances of the same node class. |
| [`xcos_loop_resolving`](xcos_loop_resolving/) | Algebraic-loop rejection and phase-based resolution of a Scilab/Xcos-style diagram. |
| [`compile_checks`](compile_checks/) | Minimal positive and negative examples for C1, C3, and C2* compile checks. |
