from __future__ import annotations

from typing import Any, cast

from regelum import (
    Goto,
    Input,
    Node,
    NodeInputs,
    NodeOutputs,
    Output,
    OutputPort,
    Phase,
    PhasedReactiveSystem,
    terminate,
)

JsonDoc = dict[str, Any]


class JsonSource(Node):
    class Outputs(NodeOutputs):
        doc: JsonDoc = Output(initial=lambda: {})

    def run(self) -> Outputs:
        return self.Outputs(
            doc={
                "request_id": "r-001",
                "raw": {"text": "hello"},
            }
        )


class FeatureEnricher(Node):
    class Inputs(NodeInputs):
        doc: JsonDoc = Input(source=lambda: JsonSource.Outputs.doc)

    class Outputs(NodeOutputs):
        doc: JsonDoc = Output(initial=lambda: {})

    def run(self, inputs: Inputs) -> Outputs:
        doc = inputs.doc
        raw = doc["raw"]
        if not isinstance(raw, dict):
            raise TypeError("Expected raw payload to be a dict.")
        text = raw["text"]
        if not isinstance(text, str):
            raise TypeError("Expected raw text to be a string.")
        doc["features"] = {"length": len(text), "uppercase": text.upper()}
        return self.Outputs(doc=doc)


class DecisionEnricher(Node):
    class Inputs(NodeInputs):
        doc: JsonDoc = Input(source=lambda: FeatureEnricher.Outputs.doc)

    class Outputs(NodeOutputs):
        doc: JsonDoc = Output(initial=lambda: {})

    def run(self, inputs: Inputs) -> Outputs:
        doc = inputs.doc
        features = doc["features"]
        if not isinstance(features, dict):
            raise TypeError("Expected features payload to be a dict.")
        length = features["length"]
        if not isinstance(length, int):
            raise TypeError("Expected feature length to be an int.")
        doc["decision"] = {"label": "short" if length < 8 else "long"}
        return self.Outputs(doc=doc)


def build_system() -> PhasedReactiveSystem:
    source = JsonSource()
    feature = FeatureEnricher()
    decision = DecisionEnricher()
    return PhasedReactiveSystem(
        phases=[
            Phase(
                "enrich",
                nodes=(source, feature, decision),
                transitions=(Goto(terminate),),
                is_initial=True,
            )
        ],
    )


def main() -> None:
    system = build_system()
    system.step()
    source_doc_port = cast(OutputPort[JsonDoc], JsonSource.Outputs.doc)
    final_doc_port = cast(OutputPort[JsonDoc], DecisionEnricher.Outputs.doc)
    source_doc = system.read(source_doc_port)
    final_doc = system.read(final_doc_port)
    print(f"same_object={source_doc is final_doc}")
    print(final_doc)


if __name__ == "__main__":
    main()
