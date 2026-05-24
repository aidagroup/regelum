import math
from typing import cast

import pytest

import regelum as rg
from tests.core.ode._support import (
    BadMathNode,
    DirectDstateInputsNode,
    DirectLazyDstateInputsNode,
    InputsNode,
    InputsStateTimeNode,
    InputsTimeNode,
    LazyDstateInputSource,
    ReorderedInputsStateTimeNode,
    StateNode,
    SwitchNode,
    TimeNode,
    TrigNode,
    TypedOnlyInputsStateNode,
)


def test_casadi_backend_reuses_graph_with_new_parameters() -> None:
    node = SwitchNode()
    system = rg.ODESystem(nodes=(node,), dt="0.1")

    system.update(state_snapshot={"Clock.time": 0.0}, time_start=0.0, time_stop=0.1)
    assert cast(SwitchNode.State, node.state()).x == pytest.approx(0.1)

    system.update(state_snapshot={"Clock.time": 1.0}, time_start=0.1, time_stop=0.2)
    assert cast(SwitchNode.State, node.state()).x == pytest.approx(0.0)


def test_casadi_backend_traces_casadi_primitives() -> None:
    node = TrigNode()
    system = rg.ODESystem(nodes=(node,), dt="0.01")

    system.update(state_snapshot={}, time_start=0.0, time_stop=0.01)

    assert cast(TrigNode.State, node.state()).x < 1.0


def test_casadi_backend_reports_untraceable_dstate() -> None:
    node = BadMathNode()
    system = rg.ODESystem(nodes=(node,), dt="0.01")

    with pytest.raises(rg.CasadiTraceError, match="casadi primitives"):
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
    system = rg.ODESystem(
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
