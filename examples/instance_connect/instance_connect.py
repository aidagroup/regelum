from __future__ import annotations

import regelum as rg


class NumberSource(rg.Node):
    class State(rg.NodeState):
        value: int = rg.var(init=0)

    def __init__(self, value: int, *, name: str | None = None) -> None:
        super().__init__(name=name)
        self.value = value

    def update(self) -> State:
        return self.State(value=self.value)


class Accumulator(rg.Node):
    class Inputs(rg.NodeInputs):
        value: int = rg.src()
        total: int = rg.src()

    class State(rg.NodeState):
        total: int = rg.var(init=0)

    def update(self, inputs: Inputs) -> State:
        return self.State(total=inputs.total + inputs.value)


def build_system() -> rg.PhasedReactiveSystem:
    source_a = NumberSource(3, name="source_a")
    source_b = NumberSource(7, name="source_b")
    accumulator_a = Accumulator(name="accumulator_a")
    accumulator_b = Accumulator(name="accumulator_b")

    rg.port(accumulator_a.Inputs.value).connect(source_a.State.value)
    rg.port(accumulator_a.State.total).connect(accumulator_a.Inputs.total)
    rg.port(source_b.State.value).connect(accumulator_b.Inputs.value)
    rg.port(accumulator_b.Inputs.total).connect(accumulator_b.State.total)

    return rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "accumulate",
                nodes=(source_a, source_b, accumulator_a, accumulator_b),
                transitions=(rg.Goto(rg.terminate),),
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
