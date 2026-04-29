from regelum import Input, Node, NodeInputs, NodeOutputs, Output
from regelum.examples.pendulum import PDController, PendulumPlant


def check_inputs(inputs: PDController.Inputs) -> None:
    theta: float = inputs.theta
    omega: float = inputs.omega
    _: tuple[float, float] = (theta, omega)


def check_sources() -> None:
    theta_source: float = PendulumPlant.Outputs.theta
    torque_source: float = PDController.Outputs.torque
    _: tuple[float, float] = (theta_source, torque_source)


class CustomSource(Node):
    class Vars(NodeOutputs):
        value: int = Output(initial=1)

    def run(self) -> Vars:
        return self.Vars(value=2)


class CustomSink(Node):
    class In(NodeInputs):
        value: int = Input(source=lambda: CustomSource.Vars.value)

    class Out(NodeOutputs):
        seen: int = Output()

    def run(self, inputs: In) -> Out:
        return self.Out(seen=inputs.value)


def check_custom_namespaces(source: CustomSource, sink: CustomSink) -> None:
    source_value: int = CustomSource.Vars.value
    sink_input: int = sink.In.value
    source_output: int = source.Vars.value
    _: tuple[int, int, int] = (source_value, sink_input, source_output)
