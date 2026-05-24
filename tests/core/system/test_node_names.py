import pytest

import regelum as rg
from tests.core.system._support import _tick_system


def test_node_name_can_default_from_class_or_instance_override() -> None:
    class NamedSource(rg.Node):
        name = "class_named_source"

        class State(rg.NodeState):
            value: int = rg.var(init=1)

    class PlainSource(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=2)

    system = _tick_system(
        [
            NamedSource(),
            PlainSource(name="instance_named_source"),
        ]
    )

    assert system.compile_report.nodes == (
        "class_named_source",
        "instance_named_source",
    )
    assert system.snapshot() == {
        "class_named_source.value": 1,
        "instance_named_source.value": 2,
    }


def test_implicit_node_names_are_deduplicated() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=1)

    system = _tick_system([Source(), Source()])

    assert system.compile_report.nodes == ("Source", "Source_2")
    assert system.snapshot() == {
        "Source.value": 1,
        "Source_2.value": 1,
    }


def test_explicit_duplicate_node_names_are_rejected() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=1)

    with pytest.raises(rg.CompileError) as exc_info:
        _tick_system([Source(name="source"), Source(name="source")])

    assert any(
        issue.location == "source" and issue.message == "node name is declared more than once"
        for issue in exc_info.value.report.issues
    )


def test_class_level_source_is_rejected_when_multiple_instances_exist() -> None:
    class Plant(rg.Node):
        class State(rg.NodeState):
            theta: float = rg.var(init=0.0)

    class Controller(rg.Node):
        class Inputs(rg.NodeInputs):
            theta: float = rg.src(Plant.State.theta)

        class State(rg.NodeState):
            torque: float = rg.var(init=0.0)

    with pytest.raises(rg.CompileError) as exc_info:
        _tick_system([Plant(), Plant(), Controller()])

    assert any(
        issue.location == "Controller.theta"
        and "ambiguous input source 'Plant.theta'" in issue.message
        and "Plant_2.theta" in issue.message
        and "use instance connection" in issue.message
        for issue in exc_info.value.report.issues
    )
