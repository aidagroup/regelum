from __future__ import annotations

import regelum as rg


def build_system() -> rg.PhasedReactiveSystem:
    return rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "a",
                nodes=(),
                transitions=(rg.Goto("b"),),
                is_initial=True,
            ),
            rg.Phase(
                "b",
                nodes=(),
                transitions=(rg.Goto("a"),),
            ),
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
