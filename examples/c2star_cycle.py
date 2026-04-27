from __future__ import annotations

from regelum import (
    CompileError,
    If,
    Node,
    NodeOutputs,
    Output,
    Phase,
    PhasedReactiveSystem,
    V,
    terminate,
)


class Mode(Node):
    class Outputs(NodeOutputs):
        flag: bool = Output(initial=False)


def build_dead_cycle_system() -> PhasedReactiveSystem:
    mode = Mode()
    return PhasedReactiveSystem(
        phases=[
            Phase(
                "a",
                nodes=(mode,),
                transitions=(
                    If(V(Mode.Outputs.flag), "b", name="to-b"),
                    If(~V(Mode.Outputs.flag), terminate, name="stop-a"),
                ),
                is_initial=True,
            ),
            Phase(
                "b",
                nodes=(mode,),
                transitions=(
                    If(~V(Mode.Outputs.flag), "a", name="to-a"),
                    If(V(Mode.Outputs.flag), terminate, name="stop-b"),
                ),
            ),
        ],
    )


def build_live_cycle_system() -> PhasedReactiveSystem:
    mode = Mode()
    return PhasedReactiveSystem(
        phases=[
            Phase(
                "a",
                nodes=(mode,),
                transitions=(
                    If(V(Mode.Outputs.flag), "b", name="to-b"),
                    If(~V(Mode.Outputs.flag), terminate, name="stop-a"),
                ),
                is_initial=True,
            ),
            Phase(
                "b",
                nodes=(mode,),
                transitions=(
                    If(V(Mode.Outputs.flag), "a", name="to-a"),
                    If(~V(Mode.Outputs.flag), terminate, name="stop-b"),
                ),
            ),
        ],
    )


def main() -> None:
    try:
        dead = build_dead_cycle_system()
    except CompileError as exc:
        print("dead cycle: compile=False")
        for issue in exc.report.issues:
            print(f"{issue.location}: {issue.message}")
    else:
        print(f"dead cycle: compile={dead.compile_report.ok}")

    try:
        build_live_cycle_system()
    except CompileError as exc:
        print("live cycle: compile=False")
        for issue in exc.report.issues:
            print(f"{issue.location}: {issue.message}")
        return
    print("live cycle: compile=True")


if __name__ == "__main__":
    main()
