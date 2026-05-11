from __future__ import annotations

from regelum import (
    CompileError,
    Goto,
    Input,
    Node,
    NodeInputs,
    NodeState,
    Phase,
    PhasedReactiveSystem,
    Var,
    terminate,
)


class First(Node):
    class Inputs(NodeInputs):
        b: float = Input(src="Second.State.b")

    class State(NodeState):
        a: float = Var(init=0.0)

    def update(self, inputs: Inputs) -> State:
        return self.State(a=inputs.b + 1.0)


class Second(Node):
    class Inputs(NodeInputs):
        a: float = Input(src=First.State.a)

    class State(NodeState):
        b: float = Var(init=0.0)

    def update(self, inputs: Inputs) -> State:
        return self.State(b=inputs.a + 1.0)


def build_system() -> PhasedReactiveSystem:
    first = First()
    second = Second()
    return PhasedReactiveSystem(
        phases=[
            Phase(
                "coupled",
                nodes=(first, second),
                transitions=(Goto(terminate),),
                is_initial=True,
            )
        ],
    )


def main() -> None:
    try:
        build_system()
    except CompileError as exc:
        for issue in exc.report.issues:
            print(f"{issue.location}: {issue.message}")
        return
    raise RuntimeError("Expected C1 violation, but system compiled.")


if __name__ == "__main__":
    main()
