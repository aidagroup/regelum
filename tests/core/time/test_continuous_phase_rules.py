import pytest

import regelum as rg
from tests.core.time._support import (
    ConstantIntegrator,
    CoupledSinkIntegrator,
    CoupledSourceIntegrator,
    ScheduledCounter,
)


def test_ode_node_rejects_instance_dt() -> None:
    with pytest.raises(TypeError, match="set dt on ODESystem"):
        ConstantIntegrator(dt="0.1")


def test_ode_node_rejects_class_dt() -> None:
    with pytest.raises(TypeError, match="set dt on ODESystem"):

        class BadIntegrator(rg.ODENode):
            dt = "0.1"

            class State(rg.NodeState):
                x: float = rg.var(init=0.0)

            def dstate(self, state: State) -> State:  # ty: ignore[invalid-method-override]
                return self.State(x=state.x)


def test_phase_cannot_mix_ode_system_and_ordinary_nodes() -> None:
    ode = rg.ODESystem(nodes=(ConstantIntegrator(),), dt="0.1")

    with pytest.raises(rg.CompileError) as exc_info:
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "mixed",
                    nodes=(ode, ScheduledCounter()),
                    transitions=(rg.Goto(rg.terminate),),
                    is_initial=True,
                )
            ]
        )

    assert any("cannot mix ODESystem" in issue.message for issue in exc_info.value.report.issues)


def test_prs_allows_only_one_continuous_phase_for_now() -> None:
    first = rg.ODESystem(nodes=(ConstantIntegrator(name="first"),), dt="0.1")
    second = rg.ODESystem(nodes=(ConstantIntegrator(name="second"),), dt="0.1")

    with pytest.raises(rg.CompileError) as exc_info:
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "first", nodes=(first,), transitions=(rg.Goto("second"),), is_initial=True
                ),
                rg.Phase("second", nodes=(second,), transitions=(rg.Goto(rg.terminate),)),
            ]
        )

    assert any("at most one phase" in issue.message for issue in exc_info.value.report.issues)


def test_independent_ode_systems_can_share_one_continuous_phase() -> None:
    first_ode = rg.ODESystem(nodes=(ConstantIntegrator(name="first"),), dt="0.1")
    second_ode = rg.ODESystem(nodes=(ConstantIntegrator(name="second"),), dt="0.1")

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "plant",
                nodes=(first_ode, second_ode),
                transitions=(rg.Goto(rg.terminate),),
                is_initial=True,
            )
        ]
    )

    system.step()

    assert system.read("first.x") == pytest.approx(0.1)
    assert system.read("second.x") == pytest.approx(0.1)


def test_continuous_phase_rejects_coupled_ode_systems() -> None:
    source_ode = rg.ODESystem(nodes=(CoupledSourceIntegrator(),), dt="0.1")
    sink_ode = rg.ODESystem(nodes=(CoupledSinkIntegrator(),), dt="0.1")

    with pytest.raises(rg.CompileError) as exc_info:
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "plant",
                    nodes=(source_ode, sink_ode),
                    transitions=(rg.Goto(rg.terminate),),
                    is_initial=True,
                )
            ]
        )

    assert any(
        "Put continuously coupled ODENodes into the same ODESystem" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_ode_system_rejects_unlisted_ode_node_dependency() -> None:
    source = CoupledSourceIntegrator(name="source")

    class BoundSinkIntegrator(rg.ODENode):
        class Inputs(rg.NodeInputs):
            source_x: float = rg.src(source.State.x)

        class State(rg.NodeState):
            x: float = rg.var(init=0.0)

        def dstate(self, inputs: Inputs) -> State:  # ty: ignore[invalid-method-override]
            return self.State(x=inputs.source_x)

    sink_ode = rg.ODESystem(nodes=(BoundSinkIntegrator(name="sink"),), dt="0.1")

    with pytest.raises(rg.CompileError) as exc_info:
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "plant",
                    nodes=(sink_ode,),
                    transitions=(rg.Goto(rg.terminate),),
                    is_initial=True,
                )
            ]
        )

    assert any(
        "source is not assigned to any phase" in issue.message
        for issue in exc_info.value.report.issues
    )
