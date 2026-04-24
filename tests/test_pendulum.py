from __future__ import annotations

from regelum.core import (
    Goto,
    Input,
    Node,
    NodeInputs,
    NodeOutputs,
    Output,
    Phase,
    ReactiveSystem,
    V,
)


class Plant(Node):
    class Outputs(NodeOutputs):
        theta: float = Output()

    def run(self, theta: float = Input(source="Plant.theta")):
        theta = theta + 1.0
        return {"theta": theta}


class Controller(Node):
    class Inputs(NodeInputs):
        theta: float = Input(source="Plant.theta")

    class Outputs(NodeOutputs):
        torque: float = Output()

    def run(self, inputs):
        return {"torque": -inputs.theta}


class SugarController(Node):
    def run(self, theta: float = Input(source=lambda: "theta")):
        return {"torque": theta * 2.0}


def test_phase_execution_uses_declared_ports() -> None:
    system = ReactiveSystem(
        phases=(
            Phase("plant", nodes=(Plant(),), transitions=(Goto("control"),), is_initial=True),
            Phase("control", nodes=(Controller(),), transitions=(Goto("plant"),)),
        ),
        initial_state={"Plant.theta": 0.0},
    )

    step_1 = system.step()
    assert step_1["Plant.theta"] == 1.0

    step_2 = system.step()
    assert step_2["Controller.torque"] == -1.0


def test_source_paths_are_normalized() -> None:
    expr = V("Plant/theta")
    assert expr.evaluate({"Plant.theta": 3.0}) == 3.0


def test_inputs_can_be_declared_in_run_signature() -> None:
    system = ReactiveSystem(nodes=(SugarController(),), initial_state={"theta": 1.5})
    snapshot = system.step()
    assert snapshot["SugarController.torque"] == 3.0
