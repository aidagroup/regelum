# Tick termination with C2*(n)

A tick may execute several phases before reaching `rg.terminate`. A cycle in
its phase graph is allowed if the compiler can prove that execution cannot stay
in it forever. **C2*(n)** is Regelum's sufficient termination check for such graphs.
It analyzes whole strongly connected components (SCCs), including paths that
switch between different cycles.

A successful termination check is one part of compilation. Phase-local
scheduling (C1), unique and exhaustive transition selection (C3), and the other
[compilation checks](compilation.md) must also succeed.

## What is checked?

An SCC is a maximal group of phases in which every phase can reach every other
by following directed edges. For `A <-> B -> C`, the cyclic SCC is `{A, B}`.
For `A <-> B <-> C -> D`, it is `{A, B, C}`. A single phase is checked only if
it has an edge to itself. The compiler finds these components from the phase
graph; it does not enumerate simple cycles.

For each cyclic component S, let P_S(n) mean:

> There exists a permitted path of n transitions that stays entirely inside S.

Then **C2*(n) holds when P_S(n) is unsatisfiable for every cyclic component**.
The path may start at any phase and domain-valid state inside the component.
It is not restricted to states reachable from the initial phase.

Here n counts **transitions**, not phases visited, ticks, or complete turns
around a cycle. A path of n transitions contains n+1 phase-entry states.
If C2*(n) holds, it also holds for any larger n: a longer path would contain
a forbidden n-transition prefix.

Every infinite path in a finite phase graph eventually stays inside one SCC.
Consequently, excluding sufficiently long internal paths in every cyclic SCC
proves tick termination, assuming C1 and C3. Testing cycles separately would
miss an execution that alternates between cycles.

## How the SMT formula works

Z3 chooses a phase and state for every step. At each transition the formula
requires that:

1. Variables owned by nodes active in the current phase may receive any values
   in their declared domains.
2. All other state variables retain their previous values.
3. An effective outgoing guard, evaluated **after** the update, selects the
   next phase inside the same SCC.

Effective guards include the priority semantics of `If`, `ElseIf`, and `Else`.
Routes are represented by symbolic phase variables and disjunctions; the
compiler does not generate a separate solver invocation for each route.
Only guard-relevant state is needed in this encoding. Guard variables that no
phase inside the component writes remain fixed throughout the path.

The check does **not** inspect or execute Python update bodies. For example,
`return self.State(value=value - 1)` alone does not establish a decreasing
counter for the checker. Progress must follow from the declared domains,
which phases can write each variable, and the guards. This conservative model
covers all node updates allowed by those declarations.

## Depth budget and finite stopping bound

The compiler checks n=1, 2, 3, ... incrementally, reusing the solver. It stops
checking a component as soon as UNSAT proves termination for that component.
Different components can pass at different depths.

For a component S, R_S contains variables that are both:

- written by a phase in S;
- read by an effective guard of an internal transition in S.

If all these variables have finite domains, the stopping bound is:

```text
N_S = number of phases in S × product of the domain sizes of variables in R_S
```

The empty product is 1. For example, three phases and two relevant Boolean
variables give N_S=3×2×2=12. Read-only guard variables do not multiply this
bound: their values stay fixed along a path. Finite domains can be declared
with `rg.var(init=0, domain=(0, 1, 2))`; Boolean and enum state also have finite
domains.

At N_S transitions, a feasible path repeats a phase and its relevant writable
values. Under the checker's update model, the repeated segment can be replayed
forever. Searching beyond N_S therefore cannot turn SAT into UNSAT.

| Constructor option | Default | Meaning |
| --- | --- | --- |
| `c2star_depth` | `None` | Optional positive transition-depth limit per SCC. |
| `c2star_max_depth` | `64` | Positive maximum transition-depth budget per SCC. |
| `max_phase_steps` | `64` | Separate runtime limit on phase execution steps within a tick. |

The checker uses the smallest applicable limit: `c2star_depth`,
`c2star_max_depth`, and N_S when available. Setting `c2star_depth=128` alone
does not override the default maximum of 64; raise `c2star_max_depth` too.
These are depth budgets, **not solver timeouts**. One solver call may still be
expensive. Increasing a compile-time budget does not change `max_phase_steps`.

When N_S is unavailable or large, an earlier UNSAT still proves termination.
There is no need to reach the full bound to pass.

## Understand the result

| Result for one SCC | Meaning | Next action |
| --- | --- | --- |
| UNSAT at any checked depth | No path of that length exists; this SCC is certified. | Continue checking other SCCs. |
| SAT below N_S, or without a finite bound | A finite internal path exists; termination remains unproved. | Increase the depth budget if appropriate, or simplify the model. |
| SAT at N_S | Infinite local residence is possible from some domain-valid entry state. | Inspect guards and state ownership; increasing depth cannot help. |
| UNKNOWN | Z3 could not decide the formula. | Simplify the encoding; do not treat this as a pass. |
| Unsupported symbolic encoding or callable internal guards | This check cannot encode the component. | Express internal guards using symbolic `rg.V(...)` expressions. |

SAT at N_S is **not automatically a reachable infinite tick**: the starting
state of the local witness may be unreachable from the initial phase. This
explains why a system that terminates in practice can still fail compilation.
The checker is sufficient rather than complete for reachable tick termination.

In default strict mode, an uncertified SCC produces a `CompileError`. To
inspect issues without accepting them as a proof:

```python
system = rg.PhasedReactiveSystem(
    phases=phases,  # Your existing phase declarations.
    c2star_max_depth=128,
    strict=False,
)
for issue in system.compile_report.issues:
    print(f"{issue.location}: {issue.message}")
```

`strict=False` does not make the system verified. The whole compilation passes
only when `system.compile_report.ok` is true.

## Example: a cyclic graph that passes

In the [split-write example](compilation.md#c2-violated-but-c2-passes-a-dead-phase-cycle),
`phi0` writes x and requires `not x and not y` to enter `phi1`.
`phi1` writes y and requires `not x and y` to return to `phi0`.
On that return, y is true and `phi0` cannot change it, so another internal
transition is impossible. The longest internal path is two transitions;
C2*(3) holds. The generic bound is eight, but the checker stops at three.

Run the example from the repository root:

```sh
uv run python -m examples.compile_checks.c2star_split_writes
```

The [regression results](scc-test-results.md) also cover switching between
16 cycles, nested loops with counter resets, mutations that introduce
nontermination, and comparison with explicitly enumerated finite graphs.

## Migration from simple-cycle checking

The name C2* and the options `c2star_depth` and `c2star_max_depth` are retained.
Their depths now count internal SCC transitions, not consecutive traversals of
individual cycles. Some previously accepted models are rejected because
alternating cycles can produce infinite execution even when each simple cycle
cannot repeat consecutively. Review explicit depth settings when upgrading.
