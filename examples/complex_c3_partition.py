from __future__ import annotations

from regelum import (
    CompileError,
    Goto,
    If,
    Node,
    NodeState,
    Phase,
    PhasedReactiveSystem,
    V,
    Var,
    terminate,
)


class Diagnostics(Node):
    class State(NodeState):
        fault: bool = Var(init=False)
        degraded: bool = Var(init=False)
        operator_override: bool = Var(init=False)


class NormalMode(Node):
    class State(NodeState):
        entered: bool = Var(init=False)


class DegradedMode(Node):
    class State(NodeState):
        entered: bool = Var(init=False)


class ShutdownMode(Node):
    class State(NodeState):
        entered: bool = Var(init=False)


def build_ok_system() -> PhasedReactiveSystem:
    diagnostics = Diagnostics()
    normal = NormalMode()
    degraded = DegradedMode()
    shutdown = ShutdownMode()
    return PhasedReactiveSystem(
        phases=[
            Phase(
                "diagnose",
                nodes=(diagnostics,),
                transitions=(
                    If(
                        V("Diagnostics.fault") & ~V("Diagnostics.operator_override"),
                        "shutdown",
                        name="hard-fault",
                    ),
                    If(
                        (V("Diagnostics.degraded") | V("Diagnostics.operator_override"))
                        & ~(V("Diagnostics.fault") & ~V("Diagnostics.operator_override")),
                        "degraded",
                        name="degraded-or-override",
                    ),
                    If(
                        ~V("Diagnostics.fault")
                        & ~V("Diagnostics.degraded")
                        & ~V("Diagnostics.operator_override"),
                        "normal",
                        name="normal",
                    ),
                ),
                is_initial=True,
            ),
            Phase("normal", nodes=(normal,), transitions=(Goto(terminate),)),
            Phase("degraded", nodes=(degraded,), transitions=(Goto(terminate),)),
            Phase("shutdown", nodes=(shutdown,), transitions=(Goto(terminate),)),
        ],
    )


def build_bad_overlap_system() -> PhasedReactiveSystem:
    diagnostics = Diagnostics()
    normal = NormalMode()
    degraded = DegradedMode()
    shutdown = ShutdownMode()
    return PhasedReactiveSystem(
        phases=[
            Phase(
                "diagnose",
                nodes=(diagnostics,),
                transitions=(
                    If(
                        V("Diagnostics.fault"),
                        "shutdown",
                        name="fault",
                    ),
                    If(
                        V("Diagnostics.degraded") | V("Diagnostics.operator_override"),
                        "degraded",
                        name="degraded-or-override",
                    ),
                    If(
                        ~V("Diagnostics.degraded"),
                        "normal",
                        name="not-degraded",
                    ),
                ),
                is_initial=True,
            ),
            Phase("normal", nodes=(normal,), transitions=(Goto(terminate),)),
            Phase("degraded", nodes=(degraded,), transitions=(Goto(terminate),)),
            Phase("shutdown", nodes=(shutdown,), transitions=(Goto(terminate),)),
        ],
    )


def main() -> None:
    ok_system = build_ok_system()
    print(f"ok system: compile={ok_system.compile_report.ok}")

    try:
        build_bad_overlap_system()
    except CompileError as exc:
        print("bad system: compile=False")
        for issue in exc.report.issues:
            print(f"{issue.location}: {issue.message}")
        return
    print("bad system: compile=True")


if __name__ == "__main__":
    main()
