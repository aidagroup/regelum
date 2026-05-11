from regelum import Input, Node, NodeInputs, NodeState, Var
from regelum.examples.pendulum import PDController, PendulumPlant


def check_inputs(inputs: PDController.Inputs) -> None:
    theta: float = inputs.theta
    omega: float = inputs.omega
    _: tuple[float, float] = (theta, omega)


def check_sources() -> None:
    theta_source: float = PendulumPlant.State.theta
    torque_source: float = PDController.State.torque
    _: tuple[float, float] = (theta_source, torque_source)


class CustomSource(Node):
    class State(NodeState):
        value: int = Var(init=1)

    def update(self) -> State:
        return self.State(value=2)


class CustomSink(Node):
    class In(NodeInputs):
        value: int = Input(src=lambda: CustomSource.State.value)

    class State(NodeState):
        seen: int = Var()

    def update(self, inputs: In) -> State:
        return self.State(seen=inputs.value)


class CustomAccumulator(Node):
    class State(NodeState):
        total: int = Var(init=0)

    def update(
        self,
        value: int = Input(src=lambda: CustomSource.State.value),
        *,
        prev_state: State,
    ) -> State:
        total: int = prev_state.total
        return self.State(**{"total": total + value})


def check_custom_namespaces(source: CustomSource, sink: CustomSink) -> None:
    source_value: int = CustomSource.State.value
    sink_input: int = sink.In.value
    source_output: int = source.State.value
    _: tuple[int, int, int] = (source_value, sink_input, source_output)
