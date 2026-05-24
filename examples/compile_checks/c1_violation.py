from __future__ import annotations

import regelum as rg


class First(rg.Node):
    class Inputs(rg.NodeInputs):
        b: float = rg.src("Second.State.b")

    class State(rg.NodeState):
        a: float = rg.var(init=0.0)

    def update(self, inputs: Inputs) -> State:
        return self.State(a=inputs.b + 1.0)


class Second(rg.Node):
    class Inputs(rg.NodeInputs):
        a: float = rg.src(First.State.a)

    class State(rg.NodeState):
        b: float = rg.var(init=0.0)

    def update(self, inputs: Inputs) -> State:
        return self.State(b=inputs.a + 1.0)


def build_system() -> rg.PhasedReactiveSystem:
    first = First()
    second = Second()
    return rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "coupled",
                nodes=(first, second),
                transitions=(rg.Goto(rg.terminate),),
                is_initial=True,
            )
        ],
    )


def main() -> None:
    try:
        build_system()
    except rg.CompileError as exc:
        for issue in exc.report.issues:
            print(f"{issue.location}: {issue.message}")
        return
    raise RuntimeError("Expected C1 violation, but system compiled.")


if __name__ == "__main__":
    main()
