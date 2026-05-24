import pytest

import regelum as rg
from tests.core.node_resolution._support import (
    Flag,
    Worker,
)


def test_guard_only_node_is_present_in_inferred_node_graph() -> None:
    flag = Flag(name="flag")
    worker = Worker()

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "work",
                nodes=(worker, flag),
                transitions=(
                    rg.If(rg.V(flag.State.ready), rg.terminate, name="ready"),
                    rg.Else(rg.terminate, name="not-ready"),
                ),
                is_initial=True,
            )
        ],
    )

    assert system.compile_report.ok
    assert system.compile_report.nodes == ("Worker", "flag")
    assert tuple(node.node_id for node in system.nodes) == ("Worker", "flag")


def test_instance_guard_source_may_point_to_node_in_another_phase() -> None:
    flag = Flag(name="flag")
    worker = Worker()

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "observe",
                nodes=(flag,),
                transitions=(rg.Goto("work"),),
                is_initial=True,
            ),
            rg.Phase(
                "work",
                nodes=(worker,),
                transitions=(
                    rg.If(rg.V(flag.State.ready), rg.terminate, name="ready"),
                    rg.Else(rg.terminate, name="not-ready"),
                ),
            ),
        ],
    )

    assert system.compile_report.ok


def test_instance_guard_source_outside_all_phases_is_incomplete_graph() -> None:
    flag = Flag(name="external_flag")
    worker = Worker()

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "work",
                nodes=(worker,),
                transitions=(
                    rg.If(rg.V(flag.State.ready), rg.terminate, name="ready"),
                    rg.Else(rg.terminate, name="not-ready"),
                ),
                is_initial=True,
            )
        ],
        strict=False,
    )

    assert any(
        issue.location == "work.ready"
        and "incomplete phase graph" in issue.message
        and "external_flag.ready" in issue.message
        and "external_flag is not assigned to any phase" in issue.message
        for issue in system.compile_report.issues
    )
    assert not any(
        issue.location == "work.ready" and "unknown guard variable" in issue.message
        for issue in system.compile_report.issues
    )


def test_class_level_guard_ref_resolves_when_single_instance_exists_in_same_phase() -> None:
    flag = Flag()

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "check",
                nodes=(flag,),
                transitions=(
                    rg.If(rg.V(Flag.State.ready), rg.terminate, name="ready"),
                    rg.Else(rg.terminate, name="not-ready"),
                ),
                is_initial=True,
            )
        ],
    )

    assert system.compile_report.ok


def test_class_level_guard_ref_resolves_when_single_instance_exists_in_another_phase() -> None:
    flag = Flag()
    worker = Worker()

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "observe",
                nodes=(flag,),
                transitions=(rg.Goto("work"),),
                is_initial=True,
            ),
            rg.Phase(
                "work",
                nodes=(worker,),
                transitions=(
                    rg.If(rg.V(Flag.State.ready), rg.terminate, name="ready"),
                    rg.Else(rg.terminate, name="not-ready"),
                ),
            ),
        ],
    )

    assert system.compile_report.ok


def test_class_level_guard_ref_is_ambiguous_with_two_instances_in_one_phase() -> None:
    with pytest.raises(rg.CompileError) as exc_info:
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "check",
                    nodes=(Flag(name="flag_a"), Flag(name="flag_b")),
                    transitions=(
                        rg.If(rg.V(Flag.State.ready), rg.terminate, name="ready"),
                        rg.Else(rg.terminate, name="not-ready"),
                    ),
                    is_initial=True,
                )
            ],
        )

    assert any(
        issue.location == "check.ready"
        and "ambiguous guard variable 'Flag.ready'" in issue.message
        and "flag_a.ready" in issue.message
        and "flag_b.ready" in issue.message
        and "use instance state reference" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_class_level_guard_ref_is_ambiguous_with_two_instances_across_phases() -> None:
    with pytest.raises(rg.CompileError) as exc_info:
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "observe-a",
                    nodes=(Flag(name="flag_a"),),
                    transitions=(rg.Goto("observe-b"),),
                    is_initial=True,
                ),
                rg.Phase(
                    "observe-b",
                    nodes=(Flag(name="flag_b"),),
                    transitions=(rg.Goto("check"),),
                ),
                rg.Phase(
                    "check",
                    nodes=(Worker(),),
                    transitions=(
                        rg.If(rg.V(Flag.State.ready), rg.terminate, name="ready"),
                        rg.Else(rg.terminate, name="not-ready"),
                    ),
                ),
            ],
        )

    assert any(
        issue.location == "check.ready"
        and "ambiguous guard variable 'Flag.ready'" in issue.message
        and "flag_a.ready" in issue.message
        and "flag_b.ready" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_instance_bound_guard_ref_selects_one_of_two_instances() -> None:
    flag_a = Flag(name="flag_a")
    flag_b = Flag(name="flag_b")

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "check",
                nodes=(flag_a, flag_b),
                transitions=(
                    rg.If(rg.V(flag_b.State.ready), rg.terminate, name="ready"),
                    rg.Else(rg.terminate, name="not-ready"),
                ),
                is_initial=True,
            )
        ],
    )

    assert system.compile_report.ok


def test_instance_bound_guard_ref_selects_instance_across_phases() -> None:
    flag_a = Flag(name="flag_a")
    flag_b = Flag(name="flag_b")

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "observe-a",
                nodes=(flag_a,),
                transitions=(rg.Goto("observe-b"),),
                is_initial=True,
            ),
            rg.Phase(
                "observe-b",
                nodes=(flag_b,),
                transitions=(rg.Goto("check"),),
            ),
            rg.Phase(
                "check",
                nodes=(Worker(),),
                transitions=(
                    rg.If(rg.V(flag_b.State.ready), rg.terminate, name="ready"),
                    rg.Else(rg.terminate, name="not-ready"),
                ),
            ),
        ],
    )

    assert system.compile_report.ok


def test_complex_guard_expression_collects_all_instance_sources() -> None:
    flag_a = Flag(name="flag_a", ready=True)
    flag_b = Flag(name="flag_b", blocked=False)
    flag_c = Flag(name="flag_c", level=5)

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "check",
                nodes=(flag_a, flag_b, flag_c),
                transitions=(
                    rg.If(
                        (rg.V(flag_a.State.ready) & ~rg.V(flag_b.State.blocked))
                        | (rg.V(flag_c.State.level) > 3),
                        rg.terminate,
                        name="complex",
                    ),
                    rg.Else(rg.terminate, name="fallback"),
                ),
                is_initial=True,
            )
        ],
    )

    assert system.compile_report.ok


def test_complex_guard_expression_reports_missing_instance_source() -> None:
    flag_a = Flag(name="flag_a", ready=True)
    flag_b = Flag(name="flag_b", blocked=False)
    flag_c = Flag(name="flag_c", level=5)

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "check",
                nodes=(flag_a, flag_c),
                transitions=(
                    rg.If(
                        (rg.V(flag_a.State.ready) & ~rg.V(flag_b.State.blocked))
                        | (rg.V(flag_c.State.level) > 3),
                        rg.terminate,
                        name="complex",
                    ),
                    rg.Else(rg.terminate, name="fallback"),
                ),
                is_initial=True,
            )
        ],
        strict=False,
    )

    assert any(
        issue.location == "check.complex"
        and "incomplete phase graph" in issue.message
        and "flag_b.blocked" in issue.message
        and "flag_b is not assigned to any phase" in issue.message
        for issue in system.compile_report.issues
    )
    assert not any(
        "flag_a.ready" in issue.message or "flag_c.level" in issue.message
        for issue in system.compile_report.issues
    )


def test_elseif_guard_source_participates_in_graph_completeness() -> None:
    flag_a = Flag(name="flag_a")
    flag_b = Flag(name="flag_b")

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "check",
                nodes=(flag_a,),
                transitions=(
                    rg.If(rg.V(flag_a.State.ready), "first", name="first"),
                    rg.ElseIf(rg.V(flag_b.State.ready), "second", name="second"),
                    rg.Else(rg.terminate, name="fallback"),
                ),
                is_initial=True,
            ),
            rg.Phase("first", nodes=(), transitions=(rg.Goto(rg.terminate),)),
            rg.Phase("second", nodes=(), transitions=(rg.Goto(rg.terminate),)),
        ],
        strict=False,
    )

    assert any(
        issue.location == "check.second"
        and "incomplete phase graph" in issue.message
        and "flag_b.ready" in issue.message
        for issue in system.compile_report.issues
    )


def test_else_does_not_add_guard_sources() -> None:
    flag = Flag()

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "check",
                nodes=(flag,),
                transitions=(
                    rg.If(rg.V(flag.State.ready), "next", name="ready"),
                    rg.Else(rg.terminate, name="fallback"),
                ),
                is_initial=True,
            ),
            rg.Phase("next", nodes=(), transitions=(rg.Goto(rg.terminate),)),
        ],
    )

    assert system.compile_report.ok


def test_goto_does_not_add_guard_sources() -> None:
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "start",
                nodes=(Worker(),),
                transitions=(rg.Goto("next"),),
                is_initial=True,
            ),
            rg.Phase("next", nodes=(), transitions=(rg.Goto(rg.terminate),)),
        ],
    )

    assert system.compile_report.ok


def test_python_lambda_guard_does_not_participate_in_source_graph() -> None:
    flag = Flag(name="external_flag")

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "check",
                nodes=(Worker(),),
                transitions=(
                    rg.If(
                        lambda state: bool(state.get("external_flag.ready", False)),
                        rg.terminate,
                        name="lambda",
                    ),
                ),
                is_initial=True,
            )
        ],
        strict=False,
    )

    assert not any("external_flag" in issue.message for issue in system.compile_report.issues)
    assert not any("external_flag" in issue.message for issue in system.compile_report.warnings)
    assert flag.node_id == "external_flag"
