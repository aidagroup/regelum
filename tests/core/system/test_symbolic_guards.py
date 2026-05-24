from enum import Enum

import regelum as rg


def test_guard_variable_accepts_state_reference() -> None:
    class Mode(rg.Node):
        class State(rg.NodeState):
            flag: bool = rg.var(init=True)

        def update(self) -> State:
            return self.State(flag=True)

    mode = Mode()
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "start",
                nodes=(mode,),
                transitions=(
                    rg.If(rg.V(Mode.State.flag), rg.terminate, name="done"),
                    rg.If(~rg.V(Mode.State.flag), rg.terminate, name="skip"),
                ),
                is_initial=True,
            )
        ],
    )

    system.step()

    assert system.history[0].phase == "start"


def test_guard_expression_accepts_state_reference_compared_to_float() -> None:
    class Level(rg.Node):
        class State(rg.NodeState):
            value: float = rg.var(init=0.0)

        def update(self) -> State:
            return self.State(value=1.25)

    class High(rg.Node):
        class State(rg.NodeState):
            reached: bool = rg.var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    class Low(rg.Node):
        class State(rg.NodeState):
            reached: bool = rg.var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    level = Level()
    high = High()
    low = Low()
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "measure",
                nodes=(level,),
                transitions=(
                    rg.If(rg.V(Level.State.value) > 1.0, "high", name="high"),
                    rg.If(rg.V(Level.State.value) <= 1.0, "low", name="low"),
                ),
                is_initial=True,
            ),
            rg.Phase("high", nodes=(high,), transitions=(rg.Goto(rg.terminate),)),
            rg.Phase("low", nodes=(low,), transitions=(rg.Goto(rg.terminate),)),
        ],
    )

    system.step()

    assert [record.phase for record in system.history] == ["measure", "high"]
    assert system.snapshot()["High.reached"] is True
    assert system.snapshot()["Low.reached"] is False


def test_enum_state_vars_are_supported_in_symbolic_guards() -> None:
    class ModeValue(Enum):
        IDLE = "idle"
        ACTIVE = "active"
        FAILED = "failed"

    class Mode(rg.Node):
        class State(rg.NodeState):
            value: ModeValue = rg.var(init=ModeValue.IDLE)

        def update(self) -> State:
            return self.State(value=ModeValue.ACTIVE)

    class Active(rg.Node):
        class State(rg.NodeState):
            reached: bool = rg.var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    class Inactive(rg.Node):
        class State(rg.NodeState):
            reached: bool = rg.var(init=False)

        def update(self) -> State:
            return self.State(reached=True)

    mode = Mode()
    active = Active()
    inactive = Inactive()
    system = rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "mode",
                nodes=(mode,),
                transitions=(
                    rg.If(rg.V(Mode.State.value) == ModeValue.ACTIVE, "active", name="active"),
                    rg.If(rg.V(Mode.State.value) != ModeValue.ACTIVE, "inactive", name="inactive"),
                ),
                is_initial=True,
            ),
            rg.Phase("active", nodes=(active,), transitions=(rg.Goto(rg.terminate),)),
            rg.Phase("inactive", nodes=(inactive,), transitions=(rg.Goto(rg.terminate),)),
        ],
    )

    assert system.compile_report.ok
    assert system.snapshot()["Mode.value"] is ModeValue.IDLE

    system.step()

    assert [record.phase for record in system.history] == ["mode", "active"]
    assert system.snapshot()["Mode.value"] is ModeValue.ACTIVE
    assert system.snapshot()["Active.reached"] is True
    assert system.snapshot()["Inactive.reached"] is False
