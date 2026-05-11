from __future__ import annotations

from regelum import (
    Goto,
    Input,
    Node,
    NodeInputs,
    NodeState,
    Phase,
    PhasedReactiveSystem,
    Var,
    port,
    terminate,
)


class NumberSource(Node):
    class State(NodeState):
        value: int = Var(init=0)

    def __init__(self, value: int, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.value = value

    def update(self) -> State:
        return self.State(value=self.value)


class Accumulator(Node):
    class Inputs(NodeInputs):
        value: int = Input()
        total: int = Input()

    class State(NodeState):
        total: int = Var(init=0)

    def update(self, inputs: Inputs) -> State:
        return self.State(total=inputs.total + inputs.value)


def build_system() -> PhasedReactiveSystem:
    source_a = NumberSource(3, name="source_a")
    source_b = NumberSource(7, name="source_b")
    accumulator_a = Accumulator(name="accumulator_a")
    accumulator_b = Accumulator(name="accumulator_b")

    port(accumulator_a.Inputs.value).connect(source_a.State.value)
    port(accumulator_a.State.total).connect(accumulator_a.Inputs.total)
    port(source_b.State.value).connect(accumulator_b.Inputs.value)
    port(accumulator_b.Inputs.total).connect(accumulator_b.State.total)

    return PhasedReactiveSystem(
        phases=[
            Phase(
                "accumulate",
                nodes=(source_a, source_b, accumulator_a, accumulator_b),
                transitions=(Goto(terminate),),
                is_initial=True,
            )
        ],
    )


def main() -> None:
    system = build_system()
    print("inputs:", system.compile_report.inputs)
    print("initial:", system.snapshot())
    system.run(steps=3)
    print("after 3 ticks:", system.snapshot())


if __name__ == "__main__":
    main()
