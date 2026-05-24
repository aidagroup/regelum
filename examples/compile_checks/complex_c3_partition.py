from __future__ import annotations

import regelum as rg


class Diagnostics(rg.Node):
    class State(rg.NodeState):
        fault: bool = rg.var(init=False)
        degraded: bool = rg.var(init=False)
        operator_override: bool = rg.var(init=False)


class NormalMode(rg.Node):
    class State(rg.NodeState):
        entered: bool = rg.var(init=False)


class DegradedMode(rg.Node):
    class State(rg.NodeState):
        entered: bool = rg.var(init=False)


class ShutdownMode(rg.Node):
    class State(rg.NodeState):
        entered: bool = rg.var(init=False)


def build_ok_system() -> rg.PhasedReactiveSystem:
    diagnostics = Diagnostics()
    normal = NormalMode()
    degraded = DegradedMode()
    shutdown = ShutdownMode()
    return rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "diagnose",
                nodes=(diagnostics,),
                transitions=(
                    rg.If(
                        rg.V("Diagnostics.fault") & ~rg.V("Diagnostics.operator_override"),
                        "shutdown",
                        name="hard-fault",
                    ),
                    rg.If(
                        (rg.V("Diagnostics.degraded") | rg.V("Diagnostics.operator_override"))
                        & ~(rg.V("Diagnostics.fault") & ~rg.V("Diagnostics.operator_override")),
                        "degraded",
                        name="degraded-or-override",
                    ),
                    rg.If(
                        ~rg.V("Diagnostics.fault")
                        & ~rg.V("Diagnostics.degraded")
                        & ~rg.V("Diagnostics.operator_override"),
                        "normal",
                        name="normal",
                    ),
                ),
                is_initial=True,
            ),
            rg.Phase("normal", nodes=(normal,), transitions=(rg.Goto(rg.terminate),)),
            rg.Phase("degraded", nodes=(degraded,), transitions=(rg.Goto(rg.terminate),)),
            rg.Phase("shutdown", nodes=(shutdown,), transitions=(rg.Goto(rg.terminate),)),
        ],
    )


def build_bad_overlap_system() -> rg.PhasedReactiveSystem:
    diagnostics = Diagnostics()
    normal = NormalMode()
    degraded = DegradedMode()
    shutdown = ShutdownMode()
    return rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "diagnose",
                nodes=(diagnostics,),
                transitions=(
                    rg.If(
                        rg.V("Diagnostics.fault"),
                        "shutdown",
                        name="fault",
                    ),
                    rg.If(
                        rg.V("Diagnostics.degraded") | rg.V("Diagnostics.operator_override"),
                        "degraded",
                        name="degraded-or-override",
                    ),
                    rg.If(
                        ~rg.V("Diagnostics.degraded"),
                        "normal",
                        name="not-degraded",
                    ),
                ),
                is_initial=True,
            ),
            rg.Phase("normal", nodes=(normal,), transitions=(rg.Goto(rg.terminate),)),
            rg.Phase("degraded", nodes=(degraded,), transitions=(rg.Goto(rg.terminate),)),
            rg.Phase("shutdown", nodes=(shutdown,), transitions=(rg.Goto(rg.terminate),)),
        ],
    )


def main() -> None:
    ok_system = build_ok_system()
    print(f"ok system: compile={ok_system.compile_report.ok}")

    try:
        build_bad_overlap_system()
    except rg.CompileError as exc:
        print("bad system: compile=False")
        for issue in exc.report.issues:
            print(f"{issue.location}: {issue.message}")
        return
    print("bad system: compile=True")


if __name__ == "__main__":
    main()
