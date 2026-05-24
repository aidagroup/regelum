import pytest

import regelum as rg
from tests.core.system._support import _tick_system


def test_bare_annotations_install_default_input_and_state_ports() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            value: int

        def update(self) -> State:
            return self.State(value=5)

    class Sink(rg.Node):
        class Inputs(rg.NodeInputs):
            value: int

        class State(rg.NodeState):
            seen: int

        def update(self, inputs: Inputs) -> State:
            return self.State(seen=inputs.value)

    source = Source()
    sink = Sink()
    rg.port(sink.Inputs.value).connect(source.State.value)
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "copy",
                nodes=(sink, source),
                transitions=(rg.Goto(rg.terminate),),
                is_initial=True,
            )
        ],
    )

    assert system.compile_report.ok

    system.step()

    assert system.compile_report.inputs["Sink.value"] == "Source.value"
    assert system.snapshot()["Sink.seen"] == 5


def test_update_parameters_can_declare_inputs() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            value: int

        def update(self) -> State:
            return self.State(value=5)

    class Sink(rg.Node):
        class State(rg.NodeState):
            seen: int

        def update(
            self,
            value: int = rg.src(lambda: Source.State.value),
        ) -> State:
            return self.State(seen=value)

    source = Source()
    sink = Sink()
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "copy",
                nodes=(sink, source),
                transitions=(rg.Goto(rg.terminate),),
                is_initial=True,
            )
        ],
    )

    system.step()

    assert system.compile_report.inputs["Sink.value"] == "Source.value"
    assert system.compile_report.phase_schedules["copy"] == ("Source", "Sink")
    assert system.snapshot()["Sink.seen"] == 5


def test_update_parameter_inputs_cannot_be_mixed_with_inputs_class() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=1)

    class Sink(rg.Node):
        class Inputs(rg.NodeInputs):
            other: int = rg.src(lambda: Source.State.value)

        class State(rg.NodeState):
            seen: int = rg.var(init=0)

        def update(
            self,
            value: int = rg.src(lambda: Source.State.value),
        ) -> State:
            return self.State(seen=value)

    with pytest.raises(rg.CompileError) as exc_info:
        _tick_system([Source(), Sink()])

    assert any(
        issue.location == "Sink.update"
        and issue.message
        == "define inputs either as a NodeInputs namespace or as update(...) parameters, not both"
        for issue in exc_info.value.report.issues
    )


def test_node_can_use_custom_input_and_state_namespace_names() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=1)

        def update(self) -> State:
            return self.State(value=5)

    class Sink(rg.Node):
        class In(rg.NodeInputs):
            value: int = rg.src(lambda: Source.State.value)

        class State(rg.NodeState):
            seen: int = rg.var()

        def update(self, inputs: In) -> State:
            return self.State(seen=inputs.value)

    source = Source()
    sink = Sink()
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "tick",
                nodes=(sink, source),
                transitions=(rg.Goto(rg.terminate),),
                is_initial=True,
            )
        ],
    )

    assert hasattr(Source.State, "value")
    assert system.compile_report.phase_schedules["tick"] == ("Source", "Sink")

    system.step()

    assert system.snapshot()["Source.value"] == 5
    assert system.snapshot()["Sink.seen"] == 5


def test_instance_bound_ports_use_state_namespace() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=0)

    class Sink(rg.Node):
        class In(rg.NodeInputs):
            value: int = rg.src()

        class State(rg.NodeState):
            seen: int = rg.var(init=0)

    source = Source()
    sink = Sink()
    connection = rg.port(sink.In.value).connect(source.State.value)

    assert connection.input.path == "Sink.value"
    assert connection.source.path == "Source.value"
    assert source.State.value.path == "Source.value"


def test_string_source_paths_support_custom_state_namespace_names() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=1)

        def update(self) -> State:
            return self.State(value=8)

    class Sink(rg.Node):
        class In(rg.NodeInputs):
            value: int = rg.src("Source.State.value")

        class State(rg.NodeState):
            seen: int = rg.var()

        def update(self, inputs: In) -> State:
            return self.State(seen=inputs.value)

    source = Source()
    sink = Sink()
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "tick",
                nodes=(source, sink),
                transitions=(rg.Goto(rg.terminate),),
                is_initial=True,
            )
        ],
    )

    assert system.compile_report.inputs["Sink.value"] == "Source.value"
    system.step()
    assert system.snapshot()["Sink.seen"] == 8


def test_initial_state_supports_custom_namespace_class_and_instance_ports() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=1)

    source = Source(name="source")
    class_system = _tick_system([Source()], initial_state={Source.State.value: 5})
    instance_system = _tick_system([source], initial_state={source.State.value: 7})

    assert class_system.snapshot()["Source.value"] == 5
    assert instance_system.snapshot()["source.value"] == 7


def test_guards_support_custom_namespace_state_refs() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=2)

        def update(self) -> State:
            return self.State(value=2)

    class Sink(rg.Node):
        class State(rg.NodeState):
            reached: bool = rg.var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    source = Source()
    sink = Sink()
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "select",
                nodes=(source,),
                transitions=(
                    rg.If(rg.V(Source.State.value) > 1, "sink"),
                    rg.Else(rg.terminate),
                ),
                is_initial=True,
            ),
            rg.Phase("sink", nodes=(sink,), transitions=(rg.Goto(rg.terminate),)),
        ],
    )

    system.step()

    assert system.snapshot()["Sink.reached"] is True


def test_instance_input_source_must_be_assigned_to_a_phase() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=1)

    class Sink(rg.Node):
        class In(rg.NodeInputs):
            value: int = rg.src()

        class State(rg.NodeState):
            seen: int = rg.var(init=0)

    source = Source(name="source")
    sink = Sink(name="sink")
    rg.port(sink.In.value).connect(source.State.value)

    with pytest.raises(rg.CompileError) as exc_info:
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "tick",
                    nodes=(sink,),
                    transitions=(rg.Goto(rg.terminate),),
                    is_initial=True,
                )
            ]
        )

    assert any(
        issue.location == "sink.value"
        and "incomplete phase graph" in issue.message
        and "node source is not assigned to any phase" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_instance_guard_source_must_be_assigned_to_a_phase() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            ready: bool = rg.var(init=True)

    source = Source(name="source")

    with pytest.raises(rg.CompileError) as exc_info:
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "tick",
                    nodes=(),
                    transitions=(rg.If(rg.V(source.State.ready), rg.terminate, name="ready"),),
                    is_initial=True,
                )
            ]
        )

    assert any(
        issue.location == "tick.ready"
        and "incomplete phase graph" in issue.message
        and "node source is not assigned to any phase" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_node_subclass_inherits_port_namespace_when_not_overridden() -> None:
    class Base(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=1)

    class Child(Base):
        pass

    system = _tick_system([Child()])

    assert hasattr(Child.State, "value")
    assert system.snapshot()["Child.value"] == 1


def test_node_subclass_can_override_port_namespace() -> None:
    class Base(rg.Node):
        class State(rg.NodeState):
            base_value: int = rg.var(init=1)

    class Child(Base):
        class State(rg.NodeState):
            child_value: int = rg.var(init=2)

    system = _tick_system([Child()])

    assert system.compile_report.state_vars == ("Child.child_value",)
    assert system.snapshot()["Child.child_value"] == 2


def test_node_rejects_two_input_namespaces() -> None:
    with pytest.raises(TypeError, match="may define zero or one input namespace"):

        class Bad(rg.Node):
            class In(rg.NodeInputs):
                first: int = rg.src()

            class MoreIn(rg.NodeInputs):
                second: int = rg.src()


def test_node_rejects_three_input_namespaces() -> None:
    with pytest.raises(TypeError, match="found 3: In, MoreIn, ExtraIn"):

        class Bad(rg.Node):
            class In(rg.NodeInputs):
                first: int = rg.src()

            class MoreIn(rg.NodeInputs):
                second: int = rg.src()

            class ExtraIn(rg.NodeInputs):
                third: int = rg.src()


def test_node_rejects_two_state_namespaces() -> None:
    with pytest.raises(TypeError, match="may define zero or one state namespace"):

        class Bad(rg.Node):
            class State(rg.NodeState):
                first: int = rg.var(init=1)

            class ExtraState(rg.NodeState):
                second: int = rg.var(init=2)


def test_node_rejects_three_state_namespaces() -> None:
    with pytest.raises(TypeError, match="found 3: State, MoreState, ExtraState"):

        class Bad(rg.Node):
            class State(rg.NodeState):
                first: int = rg.var(init=1)

            class MoreState(rg.NodeState):
                second: int = rg.var(init=2)

            class ExtraState(rg.NodeState):
                third: int = rg.var(init=3)
