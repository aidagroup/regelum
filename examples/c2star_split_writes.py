"""Worked example from `prop:strict` (notes/rg_prm/paper.tex).

The phase graph contains a 2-cycle psi1 -> psi2 -> psi1 with guards

    g1 = (x == 0 and y == 0)        on psi1 -> psi2
    g2 = (x == 0 and y == 1)        on psi2 -> psi1

The cycle write-set is split: psi1 writes only x, psi2 writes only y.
Under D = {0, 1}, the cycle is feasible for exactly one traversal --
the second traversal demands y = 0 at psi1 while psi2 has just locked
y = 1 and psi1 cannot rewrite y. So C2*(2) holds (in fact C2*(4) which
is the |D|^|R_C| collapse bound from `thm:collapse`).

Node `run` implementations are independent Bernoulli draws into {0, 1}.
Under F_max the framework treats nodes adversarially anyway -- the
Bernoulli sampler is just one concrete inhabitant of F_max chosen at
runtime.
"""

from __future__ import annotations

import random

from regelum import (
    If,
    Input,
    Node,
    NodeInputs,
    NodeOutputs,
    Output,
    Phase,
    PhasedReactiveSystem,
    V,
    terminate,
)


class XWriter(Node):
    """Node n1: active only in psi1. own(x) = n1."""

    def __init__(self, p: float = 0.5, seed: int | None = None) -> None:
        self.p = p
        self._rng = random.Random(seed)

    class Inputs(NodeInputs):
        x: int = Input(source="XWriter.x")
        y: int = Input(source="YWriter.y")

    class Outputs(NodeOutputs):
        x: int = Output(initial=0, domain=(0, 1))

    def run(self, inputs: Inputs) -> Outputs:
        sample = 1 if self._rng.random() < self.p else 0
        return self.Outputs(x=sample)


class YWriter(Node):
    """Node n2: active only in psi2. own(y) = n2."""

    def __init__(self, p: float = 0.5, seed: int | None = None) -> None:
        self.p = p
        self._rng = random.Random(seed)

    class Inputs(NodeInputs):
        x: int = Input(source="XWriter.x")
        y: int = Input(source="YWriter.y")

    class Outputs(NodeOutputs):
        y: int = Output(initial=0, domain=(0, 1))

    def run(self, inputs: Inputs) -> Outputs:
        sample = 1 if self._rng.random() < self.p else 0
        return self.Outputs(y=sample)


def build_system(
    p_x: float = 0.5,
    p_y: float = 0.5,
    seed: int | None = 0,
) -> PhasedReactiveSystem:
    g1 = (V("XWriter.x") == 0) & (V("YWriter.y") == 0)
    g2 = (V("XWriter.x") == 0) & (V("YWriter.y") == 1)
    seed_y = None if seed is None else seed + 1
    x_writer = XWriter(p=p_x, seed=seed)
    y_writer = YWriter(p=p_y, seed=seed_y)
    return PhasedReactiveSystem(
        phases=[
            Phase(
                "psi1",
                nodes=(x_writer,),
                transitions=(
                    If(g1, "psi2", name="g1"),
                    If(~g1, terminate, name="exit-psi1"),
                ),
                is_initial=True,
            ),
            Phase(
                "psi2",
                nodes=(y_writer,),
                transitions=(
                    If(g2, "psi1", name="g2"),
                    If(~g2, terminate, name="exit-psi2"),
                ),
            ),
        ],
    )


def main() -> None:
    system = build_system(p_x=0.5, p_y=0.5, seed=42)
    report = system.compile_report
    print("=== compile report ===")
    print(f"  ok      = {report.ok}")
    print(f"  outputs = {report.outputs}")
    print(f"  issues  = {len(report.issues)}")
    for issue in report.issues:
        print(f"    [{issue.location}] {issue.message}")
    if not report.ok:
        return

    print()
    print("C1, C2*, C3 all pass -> structurally sound.")
    print("Cycle psi1 -> psi2 -> psi1 is dead at the second traversal.")
    print()
    print("=== runtime trace (Bernoulli draws, p = 0.5) ===")
    for tick_index in range(1, 9):
        records = system.step()
        trace = " -> ".join(record.phase for record in records) + " -> stop"
        snap = system.snapshot()
        print(f"  tick {tick_index}: {trace:<32}  x={snap['XWriter.x']}  y={snap['YWriter.y']}")


if __name__ == "__main__":
    main()
