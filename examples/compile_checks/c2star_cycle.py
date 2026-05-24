from __future__ import annotations

import regelum as rg


class Mode(rg.Node):
    class State(rg.NodeState):
        flag: bool = rg.var(init=False)


def build_dead_cycle_system() -> rg.PhasedReactiveSystem:
    mode = Mode()
    return rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "a",
                nodes=(mode,),
                transitions=(
                    rg.If(rg.V(Mode.State.flag), "b", name="to-b"),
                    rg.If(~rg.V(Mode.State.flag), rg.terminate, name="stop-a"),
                ),
                is_initial=True,
            ),
            rg.Phase(
                "b",
                nodes=(mode,),
                transitions=(
                    rg.If(~rg.V(Mode.State.flag), "a", name="to-a"),
                    rg.If(rg.V(Mode.State.flag), rg.terminate, name="stop-b"),
                ),
            ),
        ],
    )


def build_live_cycle_system() -> rg.PhasedReactiveSystem:
    mode = Mode()
    return rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "a",
                nodes=(mode,),
                transitions=(
                    rg.If(rg.V(Mode.State.flag), "b", name="to-b"),
                    rg.If(~rg.V(Mode.State.flag), rg.terminate, name="stop-a"),
                ),
                is_initial=True,
            ),
            rg.Phase(
                "b",
                nodes=(mode,),
                transitions=(
                    rg.If(rg.V(Mode.State.flag), "a", name="to-a"),
                    rg.If(~rg.V(Mode.State.flag), rg.terminate, name="stop-b"),
                ),
            ),
        ],
    )


def main() -> None:
    try:
        dead = build_dead_cycle_system()
    except rg.CompileError as exc:
        print("dead cycle: compile=False")
        for issue in exc.report.issues:
            print(f"{issue.location}: {issue.message}")
    else:
        print(f"dead cycle: compile={dead.compile_report.ok}")

    try:
        build_live_cycle_system()
    except rg.CompileError as exc:
        print("live cycle: compile=False")
        for issue in exc.report.issues:
            print(f"{issue.location}: {issue.message}")
        return
    print("live cycle: compile=True")


if __name__ == "__main__":
    main()
