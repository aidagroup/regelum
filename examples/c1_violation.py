from __future__ import annotations

from regelum import (
    CompileError,
    Goto,
    Input,
    Node,
    NodeInputs,
    NodeOutputs,
    Output,
    Phase,
    PhasedReactiveSystem,
    terminate,
)


class First(Node):
    class Inputs(NodeInputs):
        b: float = Input(source="Second.Outputs.b")

    class Outputs(NodeOutputs):
        a: float = Output(initial=0.0)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(a=inputs.b + 1.0)


class Second(Node):
    class Inputs(NodeInputs):
        a: float = Input(source=First.Outputs.a)

    class Outputs(NodeOutputs):
        b: float = Output(initial=0.0)

    def run(self, inputs: Inputs) -> Outputs:
        return self.Outputs(b=inputs.a + 1.0)


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
