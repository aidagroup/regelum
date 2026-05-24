import pytest

import regelum as rg


def test_connect_supports_instance_node_identity() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=0)

        def __init__(self, value: int, *, name: str | None = None) -> None:
            super().__init__(name=name)
            self.value = value

        def update(self) -> State:
            return self.State(value=self.value)

    class Sink(rg.Node):
        class Inputs(rg.NodeInputs):
            value: int = rg.src()

        class State(rg.NodeState):
            seen: int = rg.var(init=0)

        def update(self, inputs: Inputs) -> State:
            return self.State(seen=inputs.value)

    source_a = Source(1, name="source_a")
    source_b = Source(2, name="source_b")
    sink_a = Sink(name="sink_a")
    sink_b = Sink(name="sink_b")

    rg.port(sink_a.Inputs.value).connect(source_a.State.value)
    rg.port(source_b.State.value).connect(sink_b.Inputs.value)

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "copy",
                nodes=(source_a, source_b, sink_a, sink_b),
                transitions=(rg.Goto(rg.terminate),),
                is_initial=True,
            )
        ],
    )

    assert system.compile_report.nodes == (
        "source_a",
        "source_b",
        "sink_a",
        "sink_b",
    )
    assert system.compile_report.inputs["sink_a.value"] == "source_a.value"
    assert system.compile_report.inputs["sink_b.value"] == "source_b.value"

    system.step()
    snapshot = system.snapshot()

    assert snapshot["sink_a.seen"] == 1
    assert snapshot["sink_b.seen"] == 2


def test_connect_rejects_non_input_left_side() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=0)

    source = Source()

    with pytest.raises(TypeError, match="input port on the left side"):
        rg.port(source.State.value).connect(source.State.value)


def test_connect_rejects_non_state_right_side() -> None:
    class Sink(rg.Node):
        class Inputs(rg.NodeInputs):
            value: int = rg.src()

    sink = Sink()

    with pytest.raises(TypeError, match="state port, state reference"):
        rg.port(sink.Inputs.value).connect(sink.Inputs.value)
