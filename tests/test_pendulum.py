from __future__ import annotations

from regelum.core import (
    CompileError,
    Goto,
    Input,
    Node,
    NodeInputs,
    NodeOutputs,
    Output,
    Phase,
    PhasedReactiveSystem,
    V,
    _phase_dependency_edges,
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


class Counter(Node):
    class Outputs(NodeOutputs):
        count: int = Output(initial=1)

    def run(self, count: int = Input(source="Counter.count")):
        return {"count": count + 1}


class HistoryNode(Node):
    class Outputs(NodeOutputs):
        items: list[int] = Output(initial=lambda: [])

    def run(self, items: list[int] = Input(source="HistoryNode.items")):
        return {"items": [*items, 1]}


def test_phase_execution_uses_declared_ports() -> None:
    system = PhasedReactiveSystem(
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
    system = PhasedReactiveSystem(nodes=(SugarController(),), initial_state={"theta": 1.5})
    snapshot = system.step()
    assert snapshot["SugarController.torque"] == 3.0


def test_phase_dependency_edges_follow_source_bindings() -> None:
    phase = Phase("control", nodes=(Plant(), Controller()))
    assert _phase_dependency_edges(phase) == [("Plant", "Controller")]


def test_phase_execution_respects_topological_order() -> None:
    system = PhasedReactiveSystem(
        phases=(
            Phase("tick", nodes=(Controller(), Plant()), is_initial=True),
        ),
        initial_state={"Plant.theta": 0.0},
    )

    snapshot = system.step()
    assert snapshot["Plant.theta"] == 1.0
    assert snapshot["Controller.torque"] == -1.0


def test_compile_report_defaults_to_ok() -> None:
    system = PhasedReactiveSystem(phases=(Phase("tick", nodes=(Plant(),), is_initial=True),))
    assert system.compile_report.ok is True
    assert system.compile_report.format() == ("ok",)


def test_strict_mode_raises_compile_error() -> None:
    try:
        PhasedReactiveSystem()
    except CompileError as error:
        assert error.report.ok is False
        assert error.report.format() == ("error: system has no nodes",)
    else:
        raise AssertionError("CompileError was not raised")


def test_non_strict_mode_exposes_report() -> None:
    system = PhasedReactiveSystem(strict=False)
    assert system.compile_report.ok is False


def test_output_initial_values_seed_runtime_state() -> None:
    system = PhasedReactiveSystem(phases=(Phase("tick", nodes=(Counter(),), is_initial=True),))
    assert system.snapshot()["Counter.count"] == 1
    assert system.step()["Counter.count"] == 2


def test_reset_rebuilds_callable_initial_state() -> None:
    system = PhasedReactiveSystem(phases=(Phase("tick", nodes=(HistoryNode(),), is_initial=True),))
    assert system.step()["HistoryNode.items"] == [1]
    system.reset()
    assert system.snapshot()["HistoryNode.items"] == []


def test_run_accumulates_history_records() -> None:
    system = PhasedReactiveSystem(
        phases=(Phase("tick", nodes=(Counter(),), is_initial=True),),
    )
    system.run(steps=2)
    assert [record.phase for record in system.history] == ["tick", "tick"]
    assert [record.node for record in system.history] == ["Counter", "Counter"]
