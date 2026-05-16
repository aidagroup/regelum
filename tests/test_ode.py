import math
from typing import Any, cast

import casadi as ca
import numpy as np
import pytest

from regelum import Clock, Input, NodeInputs, NodeState, ODENode, ODESystem, Var
from regelum.ode import CasadiTraceError


class SwitchNode(ODENode):
    class Inputs(NodeInputs):
        time: float = Input(src=Clock.time)

    class State(NodeState):
        x: float = Var(init=0.0)

    def dstate(self, inputs: Inputs, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=ca.if_else(inputs.time < 0.5, 1.0, -1.0))


class TrigNode(ODENode):
    class State(NodeState):
        x: float = Var(init=1.0)

    def dstate(self, inputs: NodeInputs, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=-ca.sin(state.x))


class BadMathNode(ODENode):
    class State(NodeState):
        x: float = Var(init=1.0)

    def dstate(self, inputs: NodeInputs, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=-math.sin(state.x))


class InputsStateTimeNode(ODENode):
    class Inputs(NodeInputs):
        u: float = Input(src="u.value")

    class State(NodeState):
        x: float = Var(init=0.0)

    def dstate(  # ty: ignore[invalid-method-override]
        self,
        inputs: Inputs,
        state: State,
        time: Any,
    ) -> State:
        return self.State(x=inputs.u + time)


class ReorderedInputsStateTimeNode(ODENode):
    class Inputs(NodeInputs):
        u: float = Input(src="u.value")

    class State(NodeState):
        x: float = Var(init=0.0)

    def dstate(  # ty: ignore[invalid-method-override]
        self,
        time: Any,
        state: State,
        inputs: Inputs,
    ) -> State:
        return self.State(x=state.x * 0.0 + inputs.u + time)


class TypedOnlyInputsStateNode(ODENode):
    class Inputs(NodeInputs):
        u: float = Input(src="u.value")

    class State(NodeState):
        x: float = Var(init=0.0)

    def dstate(  # ty: ignore[invalid-method-override]
        self,
        control: Inputs,
        memory: State,
    ) -> State:
        return self.State(x=memory.x * 0.0 + control.u)


class DirectDstateInputsNode(ODENode):
    class State(NodeState):
        x: float = Var(init=0.0)

    def dstate(  # ty: ignore[invalid-method-override]
        self,
        time: Any,
        state: State,
        a: float = Input(src="a.value"),
        b: float = Input(src="b.value"),
    ) -> State:
        return self.State(x=state.x * 0.0 + a + 2.0 * b + time)


class DirectLazyDstateInputsNode(ODENode):
    class State(NodeState):
        x: float = Var(init=0.0)

    def dstate(  # ty: ignore[invalid-method-override]
        self,
        state: State,
        u: float = Input(src=lambda: LazyDstateInputSource.State.u),
    ) -> State:
        return self.State(x=state.x * 0.0 + u)


class LazyDstateInputSource(ODENode):
    class State(NodeState):
        u: float = Var(init=4.0)

    def dstate(self, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(u=0.0 * state.u)


class InputsTimeNode(ODENode):
    class Inputs(NodeInputs):
        u: float = Input(src="u.value")

    class State(NodeState):
        x: float = Var(init=0.0)

    def dstate(self, inputs: Inputs, time: Any) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=inputs.u + time)


class TimeNode(ODENode):
    class State(NodeState):
        x: float = Var(init=0.0)

    def dstate(self, time: Any) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=time)


class InputsNode(ODENode):
    class Inputs(NodeInputs):
        u: float = Input(src="u.value")

    class State(NodeState):
        x: float = Var(init=0.0)

    def dstate(self, inputs: Inputs) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=inputs.u)


class StateNode(ODENode):
    class State(NodeState):
        x: float = Var(init=1.0)

    def dstate(self, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=state.x)


def test_casadi_backend_reuses_graph_with_new_parameters() -> None:
    node = SwitchNode()
    system = ODESystem(nodes=(node,), dt="0.1")

    system.update(state_snapshot={"Clock.time": 0.0}, time_start=0.0, time_stop=0.1)
    assert cast(SwitchNode.State, node.state()).x == pytest.approx(0.1)

    system.update(state_snapshot={"Clock.time": 1.0}, time_start=0.1, time_stop=0.2)
    assert cast(SwitchNode.State, node.state()).x == pytest.approx(0.0)


def test_casadi_backend_traces_casadi_primitives() -> None:
    node = TrigNode()
    system = ODESystem(nodes=(node,), dt="0.01")

    system.update(state_snapshot={}, time_start=0.0, time_stop=0.01)

    assert cast(TrigNode.State, node.state()).x < 1.0


def test_casadi_backend_reports_untraceable_dstate() -> None:
    node = BadMathNode()
    system = ODESystem(nodes=(node,), dt="0.01")

    with pytest.raises(CasadiTraceError, match="casadi primitives"):
        system.update(state_snapshot={}, time_start=0.0, time_stop=0.01)


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

    system.update(
        state_snapshot={"u.value": 1.0, "a.value": 2.0, "b.value": 3.0},
        time_start=0.0,
        time_stop=0.1,
    )

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


class NumpyVectorNode(ODENode):
    class Inputs(NodeInputs):
        u: np.ndarray = Input(src="u.value")

    class State(NodeState):
        x: np.ndarray = Var(init=lambda: np.zeros(3))

    def dstate(self, inputs: Inputs, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=state.x + inputs.u)


class MatrixNode(ODENode):
    class State(NodeState):
        x: np.ndarray = Var(init=lambda: np.zeros((2, 2)))

    def dstate(self, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=state.x * 0.0 + 1.0)


class ListTupleVectorNode(ODENode):
    class State(NodeState):
        x: list[float] = Var(init=lambda: [0.0, 0.0])
        y: tuple[float, float] = Var(init=(1.0, 2.0))

    def dstate(self, state: State) -> State:  # ty: ignore[invalid-method-override]
        x = cast(Any, state.x)
        y = cast(Any, state.y)
        return self.State(x=x + y, y=-y)


class AbsoluteTimeNode(ODENode):
    class State(NodeState):
        x: float = Var(init=0.0)

    def dstate(self, time: Any) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=time)


def test_casadi_numeric_backend_integrates_numpy_vector_state_and_input() -> None:
    node = NumpyVectorNode()
    system = ODESystem(nodes=(node,), dt="0.1", backend="casadi", method="cvodes")

    system.update(
        state_snapshot={"u.value": np.array([1.0, 2.0, 3.0])},
        time_start=0.0,
        time_stop=0.1,
    )

    state = cast(NumpyVectorNode.State, node.state())
    assert isinstance(state.x, np.ndarray)
    assert state.x.shape == (3,)
    assert state.x == pytest.approx(np.array([0.10517092, 0.21034184, 0.31551275]))


def test_casadi_numeric_backend_preserves_matrix_shape() -> None:
    node = MatrixNode()
    system = ODESystem(nodes=(node,), dt="0.1", backend="casadi", method="cvodes")

    system.update(state_snapshot={}, time_start=0.0, time_stop=0.1)

    state = cast(MatrixNode.State, node.state())
    assert isinstance(state.x, np.ndarray)
    assert state.x.shape == (2, 2)
    assert state.x == pytest.approx(np.full((2, 2), 0.1))


def test_casadi_trace_views_list_and_tuple_as_vectors_but_restores_containers() -> None:
    node = ListTupleVectorNode()
    system = ODESystem(nodes=(node,), dt="0.1", backend="casadi", method="cvodes")

    system.update(state_snapshot={}, time_start=0.0, time_stop=0.1)

    state = cast(ListTupleVectorNode.State, node.state())
    assert isinstance(state.x, list)
    assert isinstance(state.y, tuple)
    assert state.x == pytest.approx([0.10016675, 0.2003335])
    assert state.y == pytest.approx((0.90483742, 1.80967484))


def test_casadi_backend_uses_absolute_time_start_plus_local_tau() -> None:
    node = AbsoluteTimeNode()
    system = ODESystem(nodes=(node,), dt="0.1", backend="casadi", method="cvodes")

    system.update(state_snapshot={}, time_start=1.0, time_stop=1.1)

    assert cast(AbsoluteTimeNode.State, node.state()).x == pytest.approx(0.105, rel=1e-5)


def test_ode_system_update_requires_global_time_interval() -> None:
    node = TrigNode()
    system = ODESystem(nodes=(node,), dt="0.01")

    with pytest.raises(ValueError, match="requires time_start and time_stop"):
        system.update(state_snapshot={})


def test_casadi_backend_rejects_lsoda_default_method() -> None:
    node = TrigNode()

    with pytest.raises(ValueError, match="cannot use method='LSODA'"):
        ODESystem(nodes=(node,), dt="0.01", backend="casadi")


def test_casadi_backend_rejects_reserved_time_options() -> None:
    node = TrigNode()

    with pytest.raises(ValueError, match="must not define integration time options"):
        ODESystem(
            nodes=(node,),
            dt="0.01",
            backend="casadi",
            method="cvodes",
            options={"tf": 0.01},
        )


def test_ode_input_shape_change_is_rejected_after_graph_build() -> None:
    node = NumpyVectorNode()
    system = ODESystem(nodes=(node,), dt="0.1", backend="casadi", method="cvodes")
    system.update(
        state_snapshot={"u.value": np.array([1.0, 2.0, 3.0])},
        time_start=0.0,
        time_stop=0.1,
    )

    with pytest.raises(ValueError, match="changed shape"):
        system.update(
            state_snapshot={"u.value": np.array([1.0, 2.0])},
            time_start=0.1,
            time_stop=0.2,
        )


def test_ode_state_rejects_ragged_and_rank_gt_2_shapes() -> None:
    class RaggedNode(ODENode):
        class State(NodeState):
            x: list[list[float]] = Var(init=lambda: [[0.0], [0.0, 0.0]])

        def dstate(self, state: State) -> State:  # ty: ignore[invalid-method-override]
            return self.State(x=state.x)

    with pytest.raises(ValueError, match="rectangular"):
        ODESystem(nodes=(RaggedNode(),), dt="0.1")

    class Rank3Node(ODENode):
        class State(NodeState):
            x: np.ndarray = Var(init=lambda: np.zeros((1, 1, 1)))

        def dstate(self, state: State) -> State:  # ty: ignore[invalid-method-override]
            return self.State(x=state.x)

    with pytest.raises(ValueError, match="1D, or 2D"):
        ODESystem(nodes=(Rank3Node(),), dt="0.1")
