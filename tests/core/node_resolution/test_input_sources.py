from typing import cast

import pytest

import regelum as rg
from tests.core.node_resolution._support import (
    LinkB,
    LinkC,
    Sink,
    Source,
    UnconnectedSink,
    _messages,
    _single_phase_system,
)


def test_class_level_input_ref_resolves_when_single_instance_exists() -> None:
    source = Source(value=7)
    sink = Sink()

    system = _single_phase_system(sink, source)

    assert system.compile_report.ok
    assert system.compile_report.inputs["Sink.value"] == "Source.value"
    assert system.compile_report.phase_schedules["tick"] == ("Source", "Sink")
    system.step()
    assert system.snapshot()["Sink.seen"] == 7


def test_class_level_input_ref_resolves_across_phases_when_single_instance_exists() -> None:
    source = Source(value=7)
    sink = Sink()

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "produce",
                nodes=(source,),
                transitions=(rg.Goto("consume"),),
                is_initial=True,
            ),
            rg.Phase("consume", nodes=(sink,), transitions=(rg.Goto(rg.terminate),)),
        ],
    )

    assert system.compile_report.ok
    assert system.compile_report.inputs["Sink.value"] == "Source.value"
    system.run(steps=1)
    assert system.snapshot()["Sink.seen"] == 7


def test_class_level_input_ref_is_ambiguous_with_two_instances_in_one_phase() -> None:
    with pytest.raises(rg.CompileError) as exc_info:
        _single_phase_system(Source(), Source(), Sink())

    assert any(
        "ambiguous input source 'Source.value'" in message and "use instance connection" in message
        for message in _messages(exc_info.value)
    )


def test_class_level_input_ref_is_ambiguous_with_two_instances_across_phases() -> None:
    sink = Sink()

    with pytest.raises(rg.CompileError) as exc_info:
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "produce-a",
                    nodes=(Source(name="source_a"),),
                    transitions=(rg.Goto("produce-b"),),
                    is_initial=True,
                ),
                rg.Phase(
                    "produce-b",
                    nodes=(Source(name="source_b"),),
                    transitions=(rg.Goto("consume"),),
                ),
                rg.Phase("consume", nodes=(sink,), transitions=(rg.Goto(rg.terminate),)),
            ],
        )

    assert any(
        "ambiguous input source 'Source.value'" in message
        and "source_a.value" in message
        and "source_b.value" in message
        for message in _messages(exc_info.value)
    )


def test_instance_bound_input_ref_selects_one_of_two_instances() -> None:
    source_a = Source(value=1)
    source_b = Source(value=2)
    sink = UnconnectedSink()
    rg.port(sink.Inputs.value).connect(source_b.State.value)

    system = _single_phase_system(source_a, source_b, sink)

    assert system.compile_report.ok
    assert system.compile_report.inputs["UnconnectedSink.value"] == "Source_2.value"
    system.step()
    assert system.snapshot()["UnconnectedSink.seen"] == 2


def test_instance_bound_input_ref_may_point_to_node_in_another_phase() -> None:
    source = Source(value=3)
    sink = UnconnectedSink()
    rg.port(sink.Inputs.value).connect(source.State.value)

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "produce",
                nodes=(source,),
                transitions=(rg.Goto("consume"),),
                is_initial=True,
            ),
            rg.Phase("consume", nodes=(sink,), transitions=(rg.Goto(rg.terminate),)),
        ],
    )

    assert system.compile_report.ok
    assert system.compile_report.inputs["UnconnectedSink.value"] == "Source.value"


def test_instance_bound_input_ref_to_node_outside_phases_is_incomplete_graph() -> None:
    source = Source(value=3)
    sink = UnconnectedSink()
    rg.port(sink.Inputs.value).connect(source.State.value)

    system = _single_phase_system(sink, strict=False)

    assert any(
        issue.location == "UnconnectedSink.value"
        and "incomplete phase graph" in issue.message
        and "Source.value" in issue.message
        and "Source is not assigned to any phase" in issue.message
        for issue in system.compile_report.issues
    )
    assert not any(
        issue.location == "UnconnectedSink.value" and "unknown input source" in issue.message
        for issue in system.compile_report.issues
    )


def test_input_closure_is_not_auto_completed_from_missing_producers() -> None:
    system = _single_phase_system(LinkC(), strict=False)

    assert any(
        issue.location == "LinkC.value" and "unknown input source 'LinkB.value'" in issue.message
        for issue in system.compile_report.issues
    )
    assert tuple(system.compile_report.nodes) == ("LinkC",)


def test_input_closure_checks_second_hop_when_intermediate_node_is_covered() -> None:
    system = _single_phase_system(LinkB(), LinkC(), strict=False)

    assert any(
        issue.location == "LinkB.value" and "unknown input source 'LinkA.value'" in issue.message
        for issue in system.compile_report.issues
    )
    assert not any(issue.location == "LinkC.value" for issue in system.compile_report.issues)


def test_phase_nodes_accept_instances_only() -> None:
    with pytest.raises(TypeError, match="Phase.nodes accepts node instances only"):
        rg.Phase(
            "bad",
            nodes=(cast(rg.Node, Source),),
            transitions=(rg.Goto(rg.terminate),),
            is_initial=True,
        )


def test_duplicate_implicit_names_do_not_break_instance_input_refs() -> None:
    source_a = Source(value=1)
    source_b = Source(value=2)
    sink = UnconnectedSink()
    rg.port(sink.Inputs.value).connect(source_b.State.value)

    system = _single_phase_system(source_a, source_b, sink)

    assert system.compile_report.nodes == ("Source", "Source_2", "UnconnectedSink")
    assert system.compile_report.inputs["UnconnectedSink.value"] == "Source_2.value"


def test_duplicate_explicit_names_are_rejected() -> None:
    with pytest.raises(rg.CompileError) as exc_info:
        _single_phase_system(Source(name="source"), Source(name="source"))

    assert any(
        issue.location == "source" and issue.message == "node name is declared more than once"
        for issue in exc_info.value.report.issues
    )
