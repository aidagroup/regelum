import regelum as rg
from tests.core.system._support import _tick_system


def test_initial_state_overrides_default_state_initial_values() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=1)

        def update(self) -> State:
            return self.State(value=5)

    system = _tick_system(
        [Source()],
        initial_state={Source.State.value: 3},
    )

    assert system.snapshot()["Source.value"] == 3

    system.step()

    assert system.snapshot()["Source.value"] == 5

    system.reset(initial_state={Source.State.value: 7})

    assert system.snapshot()["Source.value"] == 7


def test_initial_state_can_supply_required_state_var_without_initial() -> None:
    class Source(rg.Node):
        class Inputs(rg.NodeInputs):
            previous: int = rg.src(lambda: Source.State.value)

        class State(rg.NodeState):
            value: int

        def update(self, inputs: Inputs) -> State:
            return self.State(value=inputs.previous + 1)

    system = _tick_system(
        [Source()],
        initial_state={Source.State.value: 10},
    )

    assert system.compile_report.ok
    assert system.compile_report.required_initial_state_vars == {
        "Source.value": ("Source.previous",)
    }
    assert system.snapshot()["Source.value"] == 10

    system.step()

    assert system.snapshot()["Source.value"] == 11
