from typing import cast

import numpy as np
import pytest

import regelum as rg
from tests.core.ode._support import (
    AbsoluteTimeNode,
    ListTupleVectorNode,
    MatrixNode,
    NumpyVectorNode,
    TrigNode,
)


def test_casadi_numeric_backend_integrates_numpy_vector_state_and_input() -> None:
    node = NumpyVectorNode()
    system = rg.ODESystem(nodes=(node,), dt="0.1", backend="casadi", method="cvodes")

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
    system = rg.ODESystem(nodes=(node,), dt="0.1", backend="casadi", method="cvodes")

    system.update(state_snapshot={}, time_start=0.0, time_stop=0.1)

    state = cast(MatrixNode.State, node.state())
    assert isinstance(state.x, np.ndarray)
    assert state.x.shape == (2, 2)
    assert state.x == pytest.approx(np.full((2, 2), 0.1))


def test_casadi_trace_views_list_and_tuple_as_vectors_but_restores_containers() -> None:
    node = ListTupleVectorNode()
    system = rg.ODESystem(nodes=(node,), dt="0.1", backend="casadi", method="cvodes")

    system.update(state_snapshot={}, time_start=0.0, time_stop=0.1)

    state = cast(ListTupleVectorNode.State, node.state())
    assert isinstance(state.x, list)
    assert isinstance(state.y, tuple)
    assert state.x == pytest.approx([0.10016675, 0.2003335])
    assert state.y == pytest.approx((0.90483742, 1.80967484))


def test_casadi_backend_uses_absolute_time_start_plus_local_tau() -> None:
    node = AbsoluteTimeNode()
    system = rg.ODESystem(nodes=(node,), dt="0.1", backend="casadi", method="cvodes")

    system.update(state_snapshot={}, time_start=1.0, time_stop=1.1)

    assert cast(AbsoluteTimeNode.State, node.state()).x == pytest.approx(0.105, rel=1e-5)


def test_ode_system_update_requires_global_time_interval() -> None:
    node = TrigNode()
    system = rg.ODESystem(nodes=(node,), dt="0.01")

    with pytest.raises(ValueError, match="requires time_start and time_stop"):
        system.update(state_snapshot={})


def test_casadi_backend_rejects_lsoda_default_method() -> None:
    node = TrigNode()

    with pytest.raises(ValueError, match="cannot use method='LSODA'"):
        rg.ODESystem(nodes=(node,), dt="0.01", backend="casadi")


def test_casadi_backend_rejects_reserved_time_options() -> None:
    node = TrigNode()

    with pytest.raises(ValueError, match="must not define integration time options"):
        rg.ODESystem(
            nodes=(node,),
            dt="0.01",
            backend="casadi",
            method="cvodes",
            options={"tf": 0.01},
        )


def test_ode_input_shape_change_is_rejected_after_graph_build() -> None:
    node = NumpyVectorNode()
    system = rg.ODESystem(nodes=(node,), dt="0.1", backend="casadi", method="cvodes")
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
    class RaggedNode(rg.ODENode):
        class State(rg.NodeState):
            x: list[list[float]] = rg.var(init=lambda: [[0.0], [0.0, 0.0]])

        def dstate(self, state: State) -> State:  # ty: ignore[invalid-method-override]
            return self.State(x=state.x)

    with pytest.raises(ValueError, match="rectangular"):
        rg.ODESystem(nodes=(RaggedNode(),), dt="0.1")

    class Rank3Node(rg.ODENode):
        class State(rg.NodeState):
            x: np.ndarray = rg.var(init=lambda: np.zeros((1, 1, 1)))

        def dstate(self, state: State) -> State:  # ty: ignore[invalid-method-override]
            return self.State(x=state.x)

    with pytest.raises(ValueError, match="1D, or 2D"):
        rg.ODESystem(nodes=(Rank3Node(),), dt="0.1")
