import pytest

import regelum as rg
from tests.core.time._support import ConstantIntegrator


def test_compile_rejects_feasible_path_that_skips_continuous_phase() -> None:
    class Mode(rg.Node):
        class State(rg.NodeState):
            bypass: bool = rg.var(init=False)

        def update(self) -> State:
            return self.State(bypass=False)

    mode = Mode()
    ode = rg.ODESystem(nodes=(ConstantIntegrator(),), dt="0.1")

    with pytest.raises(rg.CompileError) as exc_info:
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "select",
                    nodes=(mode,),
                    transitions=(
                        rg.If(rg.V(mode.State.bypass), rg.terminate, name="bypass"),
                        rg.Else("plant", name="integrate"),
                    ),
                    is_initial=True,
                ),
                rg.Phase("plant", nodes=(ode,), transitions=(rg.Goto(rg.terminate),)),
            ],
        )

    assert any(
        "continuous phase contract violation" in issue.message
        and "without reaching a continuous phase" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_compile_accepts_symbolically_infeasible_path_that_skips_continuous_phase() -> None:
    class Mode(rg.Node):
        class State(rg.NodeState):
            bypass: bool = rg.var(init=False)

        def update(self) -> State:
            return self.State(bypass=False)

    mode = Mode()
    ode = rg.ODESystem(nodes=(ConstantIntegrator(),), dt="0.1")

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "select",
                nodes=(mode,),
                transitions=(
                    rg.If(
                        rg.V(mode.State.bypass) & ~rg.V(mode.State.bypass),
                        rg.terminate,
                        name="bypass",
                    ),
                    rg.Else("plant", name="integrate"),
                ),
                is_initial=True,
            ),
            rg.Phase("plant", nodes=(ode,), transitions=(rg.Goto(rg.terminate),)),
        ],
    )

    assert system.compile_report.ok


def test_compile_rejects_feasible_path_that_reaches_continuous_phase_twice() -> None:
    class Mode(rg.Node):
        class State(rg.NodeState):
            repeat: bool = rg.var(init=False)

        def update(self) -> State:
            return self.State(repeat=False)

    mode = Mode()
    ode = rg.ODESystem(nodes=(ConstantIntegrator(),), dt="0.1")

    with pytest.raises(rg.CompileError) as exc_info:
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase("plant", nodes=(ode,), transitions=(rg.Goto("select"),), is_initial=True),
                rg.Phase(
                    "select",
                    nodes=(mode,),
                    transitions=(
                        rg.If(rg.V(mode.State.repeat), "plant", name="repeat"),
                        rg.Else(rg.terminate, name="done"),
                    ),
                ),
            ],
        )

    assert any(
        "continuous phase contract violation" in issue.message and "more than once" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_continuous_contract_accepts_many_symbolic_branches_that_all_rejoin_before_plant() -> None:
    class Router(rg.Node):
        class State(rg.NodeState):
            mode: int = rg.var(init=0, domain=(0, 1, 2))
            armed: bool = rg.var(init=True)

        def update(self) -> State:
            return self.State(mode=0, armed=True)

    router = Router()
    ode = rg.ODESystem(nodes=(ConstantIntegrator(),), dt="0.1")

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "route",
                nodes=(router,),
                transitions=(
                    rg.If(rg.V(router.State.mode) == 0, "left", name="left"),
                    rg.ElseIf(rg.V(router.State.mode) == 1, "middle", name="middle"),
                    rg.Else("right", name="right"),
                ),
                is_initial=True,
            ),
            rg.Phase(
                "left",
                nodes=(),
                transitions=(
                    rg.If(rg.V(router.State.armed), "plant", name="armed"),
                    rg.Else("plant", name="not-armed"),
                ),
            ),
            rg.Phase("middle", nodes=(), transitions=(rg.Goto("plant"),)),
            rg.Phase("right", nodes=(), transitions=(rg.Goto("plant"),)),
            rg.Phase("plant", nodes=(ode,), transitions=(rg.Goto(rg.terminate),)),
        ],
    )

    assert system.compile_report.ok


def test_continuous_contract_accepts_symbolic_branches_after_plant_that_all_terminate() -> None:
    class Router(rg.Node):
        class State(rg.NodeState):
            mode: int = rg.var(init=0, domain=(0, 1, 2))
            armed: bool = rg.var(init=True)

        def update(self) -> State:
            return self.State(mode=0, armed=True)

    router = Router()
    ode = rg.ODESystem(nodes=(ConstantIntegrator(),), dt="0.1")

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase("plant", nodes=(ode,), transitions=(rg.Goto("route"),), is_initial=True),
            rg.Phase(
                "route",
                nodes=(router,),
                transitions=(
                    rg.If((rg.V(router.State.mode) == 0) & rg.V(router.State.armed), "a", name="a"),
                    rg.ElseIf(rg.V(router.State.mode) == 1, "b", name="b"),
                    rg.Else("c", name="c"),
                ),
            ),
            rg.Phase("a", nodes=(), transitions=(rg.Goto(rg.terminate),)),
            rg.Phase("b", nodes=(), transitions=(rg.Goto(rg.terminate),)),
            rg.Phase("c", nodes=(), transitions=(rg.Goto(rg.terminate),)),
        ],
    )

    assert system.compile_report.ok


def test_continuous_contract_accepts_infeasible_second_plant_branch() -> None:
    class Router(rg.Node):
        class State(rg.NodeState):
            repeat: bool = rg.var(init=False)

        def update(self) -> State:
            return self.State(repeat=False)

    router = Router()
    ode = rg.ODESystem(nodes=(ConstantIntegrator(),), dt="0.1")

    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase("prepare", nodes=(router,), transitions=(rg.Goto("plant"),), is_initial=True),
            rg.Phase("plant", nodes=(ode,), transitions=(rg.Goto("route"),)),
            rg.Phase(
                "route",
                nodes=(),
                transitions=(
                    rg.If(
                        rg.V(router.State.repeat) & ~rg.V(router.State.repeat),
                        "plant",
                        name="repeat",
                    ),
                    rg.Else(rg.terminate, name="done"),
                ),
            ),
        ]
    )

    assert system.compile_report.ok


def test_continuous_contract_rejects_havoced_guard_that_can_skip_plant() -> None:
    class Router(rg.Node):
        class State(rg.NodeState):
            bypass: bool = rg.var(init=False)

        def update(self) -> State:
            return self.State(bypass=False)

    router = Router()
    ode = rg.ODESystem(nodes=(ConstantIntegrator(),), dt="0.1")

    with pytest.raises(rg.CompileError) as exc_info:
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "route",
                    nodes=(router,),
                    transitions=(
                        rg.If(rg.V(router.State.bypass), rg.terminate, name="bypass"),
                        rg.Else("plant", name="plant"),
                    ),
                    is_initial=True,
                ),
                rg.Phase("plant", nodes=(ode,), transitions=(rg.Goto(rg.terminate),)),
            ],
        )

    assert any(
        issue.location == "route"
        and "without reaching a continuous phase" in issue.message
        and "Router.bypass" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_continuous_contract_rejects_non_symbolic_guard_before_mandatory_plant() -> None:
    ode = rg.ODESystem(nodes=(ConstantIntegrator(),), dt="0.1")

    with pytest.raises(rg.CompileError) as exc_info:
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "route",
                    nodes=(),
                    transitions=(
                        rg.If(lambda _state: True, "plant", name="python-guard"),
                        rg.Else(rg.terminate, name="fallback"),
                    ),
                    is_initial=True,
                ),
                rg.Phase("plant", nodes=(ode,), transitions=(rg.Goto(rg.terminate),)),
            ],
        )

    assert any(
        issue.location == "route.python-guard" and "non-symbolic transition guards" in issue.message
        for issue in exc_info.value.report.issues
    )


def test_continuous_contract_rejects_deep_branch_that_can_skip_plant_after_rejoin() -> None:
    class Router(rg.Node):
        class State(rg.NodeState):
            a: bool = rg.var(init=False)
            b: bool = rg.var(init=False)
            c: bool = rg.var(init=False)

        def update(self) -> State:
            return self.State(a=False, b=False, c=False)

    router = Router()
    ode = rg.ODESystem(nodes=(ConstantIntegrator(),), dt="0.1")

    with pytest.raises(rg.CompileError) as exc_info:
        rg.PhasedReactiveSystem(
            phases=[
                rg.Phase(
                    "a",
                    nodes=(router,),
                    transitions=(
                        rg.If(rg.V(router.State.a), "b1", name="b1"),
                        rg.Else("b2", name="b2"),
                    ),
                    is_initial=True,
                ),
                rg.Phase(
                    "b1",
                    nodes=(),
                    transitions=(
                        rg.If(rg.V(router.State.b), "c1", name="c1"),
                        rg.Else("c2", name="c2"),
                    ),
                ),
                rg.Phase("b2", nodes=(), transitions=(rg.Goto("c2"),)),
                rg.Phase(
                    "c1",
                    nodes=(),
                    transitions=(
                        rg.If(rg.V(router.State.c), "plant", name="plant"),
                        rg.Else(rg.terminate, name="bad"),
                    ),
                ),
                rg.Phase("c2", nodes=(), transitions=(rg.Goto("plant"),)),
                rg.Phase("plant", nodes=(ode,), transitions=(rg.Goto(rg.terminate),)),
            ],
        )

    assert any(
        "a -> b1 -> c1" in issue.location and "without reaching a continuous phase" in issue.message
        for issue in exc_info.value.report.issues
    )
