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


def build_c3_violation() -> rg.PhasedReactiveSystem:
    return rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "phi",
                nodes=(X(),),
                transitions=(
                    rg.If(~rg.V(X.State.x), rg.terminate),
                    rg.If(~rg.V(X.State.x), rg.terminate),
                ),
                is_initial=True,
            )
        ]
    )


def build_c2star_system() -> rg.PhasedReactiveSystem:
    return rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "0",
                nodes=(X(),),
                transitions=(
                    rg.If(~rg.V(X.State.x) & ~rg.V(Y.State.y), "1"),
                    rg.Else(rg.terminate),
                ),
                is_initial=True,
            ),
            rg.Phase(
                "1",
                nodes=(Y(),),
                transitions=(
                    rg.If(~rg.V(X.State.x) & rg.V(Y.State.y), "0"),
                    rg.Else(rg.terminate),
                ),
            ),
        ]
    )


def main() -> None:
    try:
        build_c3_violation()
    except rg.CompileError as exc:
        issue = exc.report.issues[0]
        print(f"{issue.location}: {issue.message}")

    system = build_c2star_system()
    print(f"compile ok = {system.compile_report.ok}")
    print("C2*(2) status = pass")


if __name__ == "__main__":
    main()
