import math
from typing import cast

import casadi as ca
import pytest

from regelum import Clock, Input, NodeInputs, NodeState, ODENode, ODESystem, StateVar
from regelum.ode import CasadiTraceError


class SwitchNode(ODENode):
    class Inputs(NodeInputs):
        time: float = Input(source=Clock.time)

    class State(NodeState):
        x: float = StateVar(initial=0.0)

    def dstate(self, inputs: Inputs, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=ca.if_else(inputs.time < 0.5, 1.0, -1.0))


class TrigNode(ODENode):
    class State(NodeState):
        x: float = StateVar(initial=1.0)

    def dstate(self, inputs: NodeInputs, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=-ca.sin(state.x))


class BadMathNode(ODENode):
    class State(NodeState):
        x: float = StateVar(initial=1.0)

    def dstate(self, inputs: NodeInputs, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=-math.sin(state.x))


def test_casadi_backend_reuses_graph_with_new_parameters() -> None:
    node = SwitchNode()
    system = ODESystem(nodes=(node,), dt="0.1")

    system.run(state_snapshot={"Clock.time": 0.0})
    assert cast(SwitchNode.State, node.state()).x == pytest.approx(0.1)

    system.run(state_snapshot={"Clock.time": 1.0})
    assert cast(SwitchNode.State, node.state()).x == pytest.approx(0.0)


def test_casadi_backend_traces_casadi_primitives() -> None:
    node = TrigNode()
    system = ODESystem(nodes=(node,), dt="0.01")

    system.run(state_snapshot={})

    assert cast(TrigNode.State, node.state()).x < 1.0


def test_casadi_backend_reports_untraceable_dstate() -> None:
    node = BadMathNode()
    system = ODESystem(nodes=(node,), dt="0.01")

    with pytest.raises(CasadiTraceError, match="casadi primitives"):
        system.run(state_snapshot={})
