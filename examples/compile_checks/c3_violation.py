from __future__ import annotations

import regelum as rg


class ModeSource(rg.Node):
    class State(rg.NodeState):
        ready: bool = rg.var(init=False)


def build_system() -> rg.PhasedReactiveSystem:
    mode = ModeSource()
    return rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "ambiguous",
                nodes=(mode,),
                transitions=(
                    rg.If(rg.V("ModeSource.ready"), rg.terminate, name="ready"),
                    rg.If(rg.V("ModeSource.ready"), rg.terminate, name="also-ready"),
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
        return
    raise RuntimeError("Expected C3 violation, but system compiled.")


if __name__ == "__main__":
    main()
