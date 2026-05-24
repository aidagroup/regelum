import pytest

import regelum as rg
from tests.core.node_resolution._support import (
    Flag,
    Worker,
)


def test_elif_is_identical_to_elseif_transition_constructor() -> None:
    transition = rg.Elif(rg.V(Flag.State.ready), rg.terminate)
    elseif_transition = rg.ElseIf(rg.V(Flag.State.ready), rg.terminate)

    assert transition.kind == elseif_transition.kind
    assert transition.name == elseif_transition.name
    assert transition.target is elseif_transition.target
    assert getattr(transition.predicate, "variables") == getattr(
        elseif_transition.predicate,
        "variables",
    )
    assert transition.kind == "elseif"
    assert transition.name == "elseif"


def test_elif_participates_in_if_chains_like_elseif() -> None:
    mode = Flag(name="mode", ready=False, level=2)
    target = Worker()

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "route",
                nodes=(mode,),
                transitions=(
                    rg.If(rg.V(mode.State.ready), "ready", name="ready"),
                    rg.Elif(rg.V(mode.State.level) == 2, "level-two", name="level-two"),
                    rg.Else(rg.terminate, name="fallback"),
                ),
                is_initial=True,
            ),
            rg.Phase("ready", nodes=(), transitions=(rg.Goto(rg.terminate),)),
            rg.Phase("level-two", nodes=(target,), transitions=(rg.Goto(rg.terminate),)),
        ],
    )

    assert system.compile_report.ok
    system.run(steps=1)
    assert [record.phase for record in system.history][-2:] == ["route", "level-two"]
    assert system.snapshot()["Worker.ran"] is True


def test_elif_can_be_mixed_with_elseif_in_the_same_chain() -> None:
    mode = Flag(name="mode", ready=False, blocked=False, level=3)
    target = Worker()

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "route",
                nodes=(mode,),
                transitions=(
                    rg.If(rg.V(mode.State.ready), "ready", name="ready"),
                    rg.Elif(rg.V(mode.State.blocked), "blocked", name="blocked"),
                    rg.ElseIf(rg.V(mode.State.level) == 3, "level-three", name="level-three"),
                    rg.Else(rg.terminate, name="fallback"),
                ),
                is_initial=True,
            ),
            rg.Phase("ready", nodes=(), transitions=(rg.Goto(rg.terminate),)),
            rg.Phase("blocked", nodes=(), transitions=(rg.Goto(rg.terminate),)),
            rg.Phase("level-three", nodes=(target,), transitions=(rg.Goto(rg.terminate),)),
        ],
    )

    assert system.compile_report.ok
    system.run(steps=1)
    assert [record.phase for record in system.history][-2:] == ["route", "level-three"]


def test_elif_requires_open_if_chain_like_elseif() -> None:
    with pytest.raises(rg.CompileError) as exc_info:
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "bad",
                    nodes=(Flag(),),
                    transitions=(
                        rg.Elif(rg.V(Flag.State.ready), rg.terminate),
                        rg.Else(rg.terminate),
                    ),
                    is_initial=True,
                )
            ],
        )

    assert any(
        issue.location == "bad.elseif" and issue.message == "ElseIf must follow If or ElseIf"
        for issue in exc_info.value.report.issues
    )


def test_elif_after_else_is_compile_error_like_elseif() -> None:
    with pytest.raises(rg.CompileError) as exc_info:
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "bad",
                    nodes=(Flag(),),
                    transitions=(
                        rg.If(rg.V(Flag.State.ready), rg.terminate),
                        rg.Else(rg.terminate),
                        rg.Elif(~rg.V(Flag.State.ready), rg.terminate),
                    ),
                    is_initial=True,
                )
            ],
        )

    assert any(
        issue.location == "bad.elseif" and issue.message == "ElseIf must follow If or ElseIf"
        for issue in exc_info.value.report.issues
    )
