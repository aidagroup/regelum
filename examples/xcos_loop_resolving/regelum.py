import regelum as rg


class Constant(rg.Node):
    constant: float = 1.0

    class State(rg.NodeState):
        value: float = rg.var(init=lambda self: self.constant)

    def update(self) -> State:
        return self.State(value=self.constant)


class Sum(rg.Node):
    class State(rg.NodeState):
        total: float = rg.var(init=0.0)

    def update(
        self,
        left: float = rg.src(Constant.State.value),
        feedback: float = rg.src(lambda: Gain.State.output),
    ) -> State:
        return self.State(total=left + feedback)


class Gain(rg.Node):
    gain: float = -1.0

    class State(rg.NodeState):
        output: float = rg.var(init=0.0)

    def update(self, signal: float = rg.src(Sum.State.total)) -> State:
        return self.State(output=self.gain * signal)


class Trash(rg.Node):
    class State(rg.NodeState):
        last: float = rg.var(init=0.0)
        samples: list[float] = rg.var(init=list)

    def update(
        self,
        value: float = rg.src(Gain.State.output),
        samples: list[float] = rg.src(lambda: Trash.State.samples),
    ) -> State:
        return self.State(last=value, samples=samples + [value])


def main() -> None:
    print("=== literal diagram ===")
    try:
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "literal-loop",
                    nodes=(Constant(), Sum(), Gain(), Trash()),
                    transitions=(rg.Goto(rg.terminate),),
                    is_initial=True,
                )
            ],
        )
    except rg.CompileError as exc:
        for issue in exc.report.issues:
            print(f"{issue.location}: {issue.message}")

    print("\n=== phase-scheduled feedback ===")
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "sum",
                nodes=(Constant(), Sum()),
                transitions=(rg.Goto("gain"),),
                is_initial=True,
            ),
            rg.Phase(
                "gain",
                nodes=(Gain(), Trash()),
                transitions=(rg.Goto(rg.terminate),),
            ),
        ],
    )
    print(f"compile ok = {system.compile_report.ok}")
    for tick in range(1, 6):
        system.step()
        snapshot = system.snapshot()
        print(
            f"tick {tick}: "
            f"sum={snapshot['Sum.total']}, "
            f"gain={snapshot['Gain.output']}, "
            f"trash={snapshot['Trash.last']}"
        )


if __name__ == "__main__":
    main()
