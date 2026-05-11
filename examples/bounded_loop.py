"""Bounded conditional loop with structural depth N.

Pattern: unroll the loop into a linear chain of N phases. Every phase
runs the same body node; each phase has two transitions -- an
early-exit guard going to terminate, and a fall-through to the next
phase. The last phase falls through to terminate.

H is acyclic, so C2 holds vacuously (and therefore C2* and C2'). The
loop body's effect on the runtime state is unconstrained -- the
upper bound N comes purely from graph topology and is independent of
node semantics.

The hyperparameter `max_steps` (= N) is supplied at build time. The
body here is an integer accumulator with an early-exit predicate
"counter >= threshold", but the same scaffold accepts any body node
and any guard.
"""

from __future__ import annotations

from regelum import (
    If,
    Input,
    Node,
    NodeInputs,
    NodeState,
    Phase,
    PhasedReactiveSystem,
    V,
    Var,
    terminate,
)


class Accumulator(Node):
    """Adds `step_size` to its own counter on every run.

    Concrete (deterministic) inhabitant of F_max; the structural
    termination guarantee does not rely on this -- under F_max the
    loop still runs at most max_steps times because H is acyclic.
    """

    def __init__(self, step_size: int = 1) -> None:
        self.step_size = step_size

    class Inputs(NodeInputs):
        counter: int = Input(src="Accumulator.counter")

    class State(NodeState):
        counter: int = Var(init=0)

    def update(self, inputs: Inputs) -> State:
        return self.State(counter=inputs.counter + self.step_size)


def build_system(
    max_steps: int,
    threshold: int,
    step_size: int = 1,
) -> PhasedReactiveSystem:
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    early_exit = V("Accumulator.counter") >= threshold
    accumulator = Accumulator(step_size=step_size)

    phases: list[Phase] = []
    for index in range(max_steps):
        is_last = index == max_steps - 1
        next_target = None if is_last else f"step_{index + 1}"
        phases.append(
            Phase(
                name=f"step_{index}",
                nodes=(accumulator,),
                transitions=(
                    If(early_exit, terminate, name="early-exit"),
                    If(~early_exit, next_target, name="continue"),
                ),
                is_initial=(index == 0),
            )
        )

    return PhasedReactiveSystem(phases=phases)


def main() -> None:
    print("=== bounded loop, max_steps=8, threshold=3 ===")
    system = build_system(max_steps=8, threshold=3)
    report = system.compile_report
    print(f"compile ok = {report.ok}, issues = {len(report.issues)}")

    records = system.step()
    trace = " -> ".join(record.phase for record in records)
    snap = system.snapshot()
    print(f"trace      = {trace} -> stop")
    print(f"final c    = {snap['Accumulator.counter']}")
    print(f"steps run  = {len(records)}  (early exit at threshold=3)")

    print()
    print("=== bounded loop, max_steps=4, threshold=999 (cap dominates) ===")
    system = build_system(max_steps=4, threshold=999)
    print(f"compile ok = {system.compile_report.ok}")
    records = system.step()
    trace = " -> ".join(record.phase for record in records)
    print(f"trace      = {trace} -> stop")
    print(f"final c    = {system.snapshot()['Accumulator.counter']}")
    print(f"steps run  = {len(records)}  (no early exit, cap = 4)")


if __name__ == "__main__":
    main()
