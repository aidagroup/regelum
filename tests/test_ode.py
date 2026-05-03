import math
from typing import Any, cast

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


class InputsStateTimeNode(ODENode):
    class Inputs(NodeInputs):
        u: float = Input(source="u.value")

    class State(NodeState):
        x: float = StateVar(initial=0.0)

    def dstate(  # ty: ignore[invalid-method-override]
        self,
        inputs: Inputs,
        state: State,
        time: Any,
    ) -> State:
        return self.State(x=inputs.u + time)


class ReorderedInputsStateTimeNode(ODENode):
    class Inputs(NodeInputs):
        u: float = Input(source="u.value")

    class State(NodeState):
        x: float = StateVar(initial=0.0)

    def dstate(  # ty: ignore[invalid-method-override]
        self,
        time: Any,
        state: State,
        inputs: Inputs,
    ) -> State:
        return self.State(x=state.x * 0.0 + inputs.u + time)


class TypedOnlyInputsStateNode(ODENode):
    class Inputs(NodeInputs):
        u: float = Input(source="u.value")

    class State(NodeState):
        x: float = StateVar(initial=0.0)

    def dstate(  # ty: ignore[invalid-method-override]
        self,
        control: Inputs,
        memory: State,
    ) -> State:
        return self.State(x=memory.x * 0.0 + control.u)


class DirectDstateInputsNode(ODENode):
    class State(NodeState):
        x: float = StateVar(initial=0.0)

    def dstate(  # ty: ignore[invalid-method-override]
        self,
        time: Any,
        state: State,
        a: float = Input(source="a.value"),
        b: float = Input(source="b.value"),
    ) -> State:
        return self.State(x=state.x * 0.0 + a + 2.0 * b + time)


class DirectLazyDstateInputsNode(ODENode):
    class State(NodeState):
        x: float = StateVar(initial=0.0)

    def dstate(  # ty: ignore[invalid-method-override]
        self,
        state: State,
        u: float = Input(source=lambda: LazyDstateInputSource.State.u),
    ) -> State:
        return self.State(x=state.x * 0.0 + u)


class LazyDstateInputSource(ODENode):
    class State(NodeState):
        u: float = StateVar(initial=4.0)

    def dstate(self, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(u=0.0 * state.u)


class InputsTimeNode(ODENode):
    class Inputs(NodeInputs):
        u: float = Input(source="u.value")

    class State(NodeState):
        x: float = StateVar(initial=0.0)

    def dstate(self, inputs: Inputs, time: Any) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=inputs.u + time)


class TimeNode(ODENode):
    class State(NodeState):
        x: float = StateVar(initial=0.0)

    def dstate(self, time: Any) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=time)


class InputsNode(ODENode):
    class Inputs(NodeInputs):
        u: float = Input(source="u.value")

    class State(NodeState):
        x: float = StateVar(initial=0.0)

    def dstate(self, inputs: Inputs) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=inputs.u)


class StateNode(ODENode):
    class State(NodeState):
        x: float = StateVar(initial=1.0)

    def dstate(self, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=state.x)


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


def test_casadi_backend_supports_dstate_argument_subsets() -> None:
    inputs_state_time = InputsStateTimeNode()
    reordered_inputs_state_time = ReorderedInputsStateTimeNode()
    typed_only_inputs_state = TypedOnlyInputsStateNode()
    direct_dstate_inputs = DirectDstateInputsNode()
    direct_lazy_dstate_inputs = DirectLazyDstateInputsNode()
    lazy_dstate_input_source = LazyDstateInputSource()
    inputs_time = InputsTimeNode()
    time_node = TimeNode()
    inputs_node = InputsNode()
    state_node = StateNode()
    system = ODESystem(
        nodes=(
            inputs_state_time,
            reordered_inputs_state_time,
            typed_only_inputs_state,
            direct_dstate_inputs,
            direct_lazy_dstate_inputs,
            lazy_dstate_input_source,
            inputs_time,
            time_node,
            inputs_node,
            state_node,
        ),
        dt="0.1",
    )

    assert DirectDstateInputsNode._inputs.keys() == {"a", "b"}
    assert DirectLazyDstateInputsNode._inputs.keys() == {"u"}

    system.run(state_snapshot={"u.value": 1.0, "a.value": 2.0, "b.value": 3.0})

    assert cast(InputsStateTimeNode.State, inputs_state_time.state()).x == pytest.approx(
        0.105,
        rel=1e-5,
    )
    assert cast(
        ReorderedInputsStateTimeNode.State,
        reordered_inputs_state_time.state(),
    ).x == pytest.approx(0.105, rel=1e-5)
    assert cast(TypedOnlyInputsStateNode.State, typed_only_inputs_state.state()).x == pytest.approx(
        0.1
    )
    assert cast(DirectDstateInputsNode.State, direct_dstate_inputs.state()).x == pytest.approx(
        0.805,
        rel=1e-5,
    )
    assert cast(
        DirectLazyDstateInputsNode.State,
        direct_lazy_dstate_inputs.state(),
    ).x == pytest.approx(0.4)
    assert cast(InputsTimeNode.State, inputs_time.state()).x == pytest.approx(
        0.105,
        rel=1e-5,
    )
    assert cast(TimeNode.State, time_node.state()).x == pytest.approx(0.005, rel=1e-5)
    assert cast(InputsNode.State, inputs_node.state()).x == pytest.approx(0.1)
    assert cast(StateNode.State, state_node.state()).x == pytest.approx(math.exp(0.1), rel=1e-4)
