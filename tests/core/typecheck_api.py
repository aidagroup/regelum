from __future__ import annotations

import regelum as rg
from examples.controlled_pendulum.standalone import Controller, PendulumODE


def check_inputs(inputs: ObserverInputs) -> None:
    theta: float = inputs.theta
    omega: float = inputs.omega
    _: tuple[float, float] = (theta, omega)


def check_sources() -> None:
    theta_source: float = PendulumODE.State.theta
    torque_source: float = Controller.State.tau
    _: tuple[float, float] = (theta_source, torque_source)


class ObserverInputs(rg.NodeInputs):
    theta: float = rg.src(PendulumODE.State.theta)
    omega: float = rg.src(PendulumODE.State.omega)


class CustomSource(rg.Node):
    class State(rg.NodeState):
        value: int = rg.var(init=1)

    def update(self) -> State:
        return self.State(value=2)


class CustomSink(rg.Node):
    class In(rg.NodeInputs):
        value: int = rg.src(lambda: CustomSource.State.value)

    class State(rg.NodeState):
        seen: int = rg.var()

    def update(self, inputs: In) -> State:
        return self.State(seen=inputs.value)


class CustomAccumulator(rg.Node):
    class State(rg.NodeState):
        total: int = rg.var(init=0)

    def update(
        self,
        value: int = rg.src(lambda: CustomSource.State.value),
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
