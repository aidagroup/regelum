from __future__ import annotations

import regelum as rg


class X(rg.Node):
    class State(rg.NodeState):
        x: bool = rg.var(init=False)


def build_system() -> rg.PhasedReactiveSystem:
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
        ],
    )


def main() -> None:
    try:
        build_system()
    except rg.CompileError as exc:
        for issue in exc.report.issues:
            print(f"{issue.location}: {issue.message}")


if __name__ == "__main__":
    main()
