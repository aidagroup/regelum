import pytest

import regelum as rg


def test_compile_rejects_multiple_initial_phases() -> None:
    with pytest.raises(rg.CompileError) as exc_info:
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase("one", nodes=(), transitions=(rg.Goto(rg.terminate),), is_initial=True),
                rg.Phase("two", nodes=(), transitions=(rg.Goto(rg.terminate),), is_initial=True),
            ],
        )

    assert not exc_info.value.report.ok
    assert any(
        issue.message == "exactly one phase must be marked initial"
        for issue in exc_info.value.report.issues
    )


def test_compile_rejects_missing_initial_phase_marker() -> None:
    with pytest.raises(rg.CompileError) as exc_info:
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase("one", nodes=(), transitions=(rg.Goto(rg.terminate),)),
                rg.Phase("two", nodes=(), transitions=(rg.Goto(rg.terminate),)),
            ],
        )

    assert not exc_info.value.report.ok
    assert any(
        issue.message == "exactly one phase must be marked initial"
        for issue in exc_info.value.report.issues
    )


def test_compile_rejects_unreachable_phase() -> None:
    with pytest.raises(rg.CompileError) as exc_info:
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase("start", nodes=(), transitions=(rg.Goto(rg.terminate),), is_initial=True),
                rg.Phase("orphan", nodes=(), transitions=(rg.Goto(rg.terminate),)),
            ],
        )

    assert any(
        issue.location == "orphan"
        and issue.message == "phase is unreachable from initial phase 'start'"
        for issue in exc_info.value.report.issues
    )


def test_compile_report_records_unreachable_phase_when_not_strict() -> None:
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase("start", nodes=(), transitions=(rg.Goto(rg.terminate),), is_initial=True),
            rg.Phase("orphan", nodes=(), transitions=(rg.Goto(rg.terminate),)),
        ],
        strict=False,
    )

    assert not system.compile_report.ok
    assert any(
        issue.location == "orphan"
        and issue.message == "phase is unreachable from initial phase 'start'"
        for issue in system.compile_report.issues
    )


def test_c2_cycle_with_local_guard_node_is_rejected_when_c2star_is_feasible() -> None:
    class Mode(rg.Node):
        class State(rg.NodeState):
            flag: bool = rg.var(init=False)

    with pytest.raises(rg.CompileError) as exc_info:
        mode = Mode()
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "a",
                    nodes=(mode,),
                    transitions=(
                        rg.If(rg.V("Mode.flag"), "b", name="to-b"),
                        rg.If(~rg.V("Mode.flag"), rg.terminate, name="stop-a"),
                    ),
                    is_initial=True,
                ),
                rg.Phase(
                    "b",
                    nodes=(mode,),
                    transitions=(
                        rg.If(~rg.V("Mode.flag"), "a", name="to-a"),
                        rg.If(rg.V("Mode.flag"), rg.terminate, name="stop-b"),
                    ),
                ),
            ],
        )

    assert any("C2*" in issue.message for issue in exc_info.value.report.issues)


def test_c2star_rejects_feasible_cycle() -> None:
    class Mode(rg.Node):
        class State(rg.NodeState):
            flag: bool = rg.var(init=False)

    with pytest.raises(rg.CompileError) as exc_info:
        mode = Mode()
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "a",
                    nodes=(mode,),
                    transitions=(
                        rg.If(rg.V("Mode.flag"), "b", name="to-b"),
                        rg.If(~rg.V("Mode.flag"), rg.terminate, name="stop-a"),
                    ),
                    is_initial=True,
                ),
                rg.Phase(
                    "b",
                    nodes=(mode,),
                    transitions=(
                        rg.If(rg.V("Mode.flag"), "a", name="to-a"),
                        rg.If(~rg.V("Mode.flag"), rg.terminate, name="stop-b"),
                    ),
                ),
            ],
        )

    assert any("C2*" in issue.message for issue in exc_info.value.report.issues)


def test_c2star_uses_effective_elseif_guards() -> None:
    class Mode(rg.Node):
        class State(rg.NodeState):
            flag: bool = rg.var(init=False)
            gate: bool = rg.var(init=False)

    with pytest.raises(rg.CompileError) as exc_info:
        mode = Mode()
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "a",
                    nodes=(mode,),
                    transitions=(
                        rg.If(rg.V("Mode.flag"), rg.terminate, name="done-a"),
                        rg.ElseIf(rg.V("Mode.gate"), "b", name="to-b"),
                        rg.Else(rg.terminate, name="stop-a"),
                    ),
                    is_initial=True,
                ),
                rg.Phase(
                    "b",
                    nodes=(mode,),
                    transitions=(
                        rg.If(rg.V("Mode.flag"), rg.terminate, name="done-b"),
                        rg.ElseIf(rg.V("Mode.gate"), "a", name="to-a"),
                        rg.Else(rg.terminate, name="stop-b"),
                    ),
                ),
            ],
        )

    assert any(
        "C2*" in issue.message and "a -> b -> a" in issue.location
        for issue in exc_info.value.report.issues
    )
