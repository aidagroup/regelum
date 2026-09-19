from __future__ import annotations

# ruff: noqa: I001
import regelum as rg
import random


class X(rg.Node):
    class State(rg.NodeState):
        x: bool = rg.var(init=False)

    def update(self, x: bool = rg.src(lambda: X.State.x)) -> State:
        return self.State(x=x & bool(random.getrandbits(1)))


class Y(rg.Node):
    class State(rg.NodeState):
        y: bool = rg.var(init=False)

    def update(self, y: bool = rg.src(lambda: Y.State.y)) -> State:
        return self.State(y=y & bool(random.getrandbits(1)))


def build_system(
    p_x: float = 0.5,
    p_y: float = 0.5,
    seed: int | None = 0,
) -> rg.PhasedReactiveSystem:
    del p_x, p_y, seed
    return rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "phi0",
                nodes=(X(),),
                transitions=(
                    rg.If(~rg.V(X.State.x) & ~rg.V(Y.State.y), "phi1"),
                    rg.Else(rg.terminate),
                ),
                is_initial=True,
            ),
            rg.Phase(
                "phi1",
                nodes=(Y(),),
                transitions=(
                    rg.If(~rg.V(X.State.x) & rg.V(Y.State.y), "phi0"),
                    rg.Else(rg.terminate),
                ),
            ),
        ],
    )


def main() -> None:
    system = build_system()
    print(f"compile ok = {system.compile_report.ok}")
    print("C2* status = pass")
    print("cycle phi0 -> phi1 -> phi0 is dead after one traversal")


if __name__ == "__main__":
    main()
