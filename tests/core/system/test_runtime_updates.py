from typing import cast

import regelum as rg
from tests.core.system._support import (
    Any,
    _tick_system,
)


def test_node_without_inputs_can_run_without_input_argument() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=lambda: 0)

        def update(self) -> State:
            return self.State(value=5)

    system = _tick_system([Source()])

    system.step()

    assert system.snapshot()["Source.value"] == 5


def test_state_var_can_be_enriched_by_reference() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            doc: dict[str, Any] = rg.var(init=lambda: {})

        def update(self) -> State:
            return self.State(
                doc={
                    "request_id": "r-001",
                    "raw": {"text": "hello"},
                }
            )

    class Enricher(rg.Node):
        class Inputs(rg.NodeInputs):
            doc: dict[str, Any] = rg.src(lambda: Source.State.doc)

        class State(rg.NodeState):
            doc: dict[str, Any] = rg.var(init=lambda: {})

        def update(self, inputs: Inputs) -> State:
            doc = inputs.doc
            raw = doc["raw"]
            assert isinstance(raw, dict)
            text = raw["text"]
            assert isinstance(text, str)
            doc["features"] = {"length": len(text)}
            return self.State(doc=doc)

    source = Source()
    enricher = Enricher()
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "enrich",
                nodes=(source, enricher),
                transitions=(rg.Goto(rg.terminate),),
                is_initial=True,
            ),
        ],
    )

    source_doc_port = cast(rg.VarPort[dict[str, Any]], Source.State.doc)
    enriched_doc_port = cast(rg.VarPort[dict[str, Any]], Enricher.State.doc)

    system.step()

    source_doc = system.read(source_doc_port)
    enriched_doc = system.read(enriched_doc_port)
    assert source_doc is enriched_doc
    assert enriched_doc["features"] == {"length": 5}


def test_input_source_accepts_lazy_state_reference() -> None:
    class Early(rg.Node):
        class Inputs(rg.NodeInputs):
            value: int = rg.src(lambda: Later.State.value)
            previous: int = rg.src(lambda: Early.State.total)

        class State(rg.NodeState):
            total: int = rg.var(init=0)

        def update(self, inputs: Inputs) -> State:
            return self.State(total=inputs.value + inputs.previous)

    class Later(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=2)

        def update(self, inputs: rg.NodeInputs) -> State:
            return self.State(value=3)

    early = Early()
    later = Later()
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase("produce", nodes=(later,), transitions=(rg.Goto("consume"),), is_initial=True),
            rg.Phase("consume", nodes=(early,), transitions=(rg.Goto(rg.terminate),)),
        ],
    )

    assert system.compile_report.inputs["Early.value"] == "Later.value"
    assert system.compile_report.inputs["Early.previous"] == "Early.total"

    system.step()

    assert system.snapshot()["Early.total"] == 3


def test_phase_runs_nodes_in_topological_order() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=0)

        def update(self) -> State:
            return self.State(value=3)

    class Sink(rg.Node):
        class Inputs(rg.NodeInputs):
            value: int = rg.src(lambda: Source.State.value)

        class State(rg.NodeState):
            seen: int = rg.var(init=0)

        def update(self, inputs: Inputs) -> State:
            return self.State(seen=inputs.value)

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

    assert [record.node for record in system.history] == ["Source", "Sink"]
    assert system.snapshot()["Sink.seen"] == 3
