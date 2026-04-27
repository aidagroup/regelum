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


class ModeSource(Node):
    class Outputs(NodeOutputs):
        ready: bool = Output(initial=False)


def build_system() -> PhasedReactiveSystem:
    mode = ModeSource()
    return PhasedReactiveSystem(
        phases=[
            Phase(
                "ambiguous",
                nodes=(mode,),
                transitions=(
                    If(V("ModeSource.ready"), terminate, name="ready"),
                    If(V("ModeSource.ready"), terminate, name="also-ready"),
                ),
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
    raise RuntimeError("Expected C3 violation, but system compiled.")


if __name__ == "__main__":
    main()
