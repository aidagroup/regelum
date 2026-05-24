import regelum as rg


class Source(rg.Node):
    class State(rg.NodeState):
        value: int = rg.var(init=1)
        ready: bool = rg.var(init=True)

    def __init__(
        self,
        value: int = 1,
        ready: bool = True,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.value = value
        self.ready = ready

    def update(self) -> State:
        return self.State(value=self.value, ready=self.ready)


class Sink(rg.Node):
    class Inputs(rg.NodeInputs):
        value: int = rg.src(Source.State.value)

    class State(rg.NodeState):
        seen: int = rg.var(init=0)

    def update(self, inputs: Inputs) -> State:
        return self.State(seen=inputs.value)


class UnconnectedSink(rg.Node):
    class Inputs(rg.NodeInputs):
        value: int = rg.src()

    class State(rg.NodeState):
        seen: int = rg.var(init=0)

    def update(self, inputs: Inputs) -> State:
        return self.State(seen=inputs.value)


class Worker(rg.Node):
    class State(rg.NodeState):
        ran: bool = rg.var(init=False)

    def update(self) -> State:
        return self.State(ran=True)


class Flag(rg.Node):
    class State(rg.NodeState):
        ready: bool = rg.var(init=True)
        blocked: bool = rg.var(init=False)
        level: int = rg.var(init=0)

    def __init__(
        self,
        *,
        ready: bool = True,
        blocked: bool = False,
        level: int = 0,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.ready = ready
        self.blocked = blocked
        self.level = level

    def update(self) -> State:
        return self.State(
            ready=self.ready,
            blocked=self.blocked,
            level=self.level,
        )


class LinkA(rg.Node):
    class State(rg.NodeState):
        value: int = rg.var(init=1)


class LinkB(rg.Node):
    class Inputs(rg.NodeInputs):
        value: int = rg.src(LinkA.State.value)

    class State(rg.NodeState):
        value: int = rg.var(init=0)

    def update(self, inputs: Inputs) -> State:
        return self.State(value=inputs.value)


class LinkC(rg.Node):
    class Inputs(rg.NodeInputs):
        value: int = rg.src(LinkB.State.value)

    class State(rg.NodeState):
        value: int = rg.var(init=0)

    def update(self, inputs: Inputs) -> State:
        return self.State(value=inputs.value)


def _single_phase_system(*nodes: rg.Node, strict: bool = True) -> rg.PhasedReactiveSystem:
    return rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "tick",
                nodes=nodes,
                transitions=(rg.Goto(rg.terminate),),
                is_initial=True,
            )
        ],
        strict=strict,
    )


def _messages(error: rg.CompileError) -> tuple[str, ...]:
    return tuple(issue.message for issue in error.report.issues)
