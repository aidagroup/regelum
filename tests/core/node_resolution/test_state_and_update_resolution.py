import pytest

import regelum as rg
from tests.core.node_resolution._support import (
    Source,
    _single_phase_system,
)


def test_var_uses_init_keyword_without_initial_alias() -> None:
    old_keyword = "initial"
    with pytest.raises(TypeError, match="unexpected keyword argument 'initial'"):
        rg.var(**{old_keyword: 0})


def test_update_can_receive_previous_state_without_declaring_self_input() -> None:
    class Counter(rg.Node):
        class State(rg.NodeState):
            count: int = rg.var(init=0)

        def update(self, prev_state: State) -> State:
            return self.State(count=prev_state.count + 1)

    counter = Counter()
    system = _single_phase_system(counter)

    assert system.compile_report.ok
    assert system.compile_report.inputs == {}
    system.run(steps=3)
    assert system.read(counter.State.count) == 3


def test_update_can_receive_inputs_object_and_previous_state() -> None:
    class Accumulator(rg.Node):
        class Inputs(rg.NodeInputs):
            value: int = rg.src(Source.State.value)

        class State(rg.NodeState):
            total: int = rg.var(init=0)

        def update(self, inputs: Inputs, state: State) -> State:
            return self.State(total=state.total + inputs.value)

    source = Source(value=2)
    accumulator = Accumulator()
    system = _single_phase_system(source, accumulator)

    assert system.compile_report.ok
    assert system.compile_report.inputs["Accumulator.value"] == "Source.value"
    system.run(steps=2)
    assert system.read(accumulator.State.total) == 4


def test_update_can_receive_previous_state_from_deferred_annotation() -> None:
    class Counter(rg.Node):
        class State(rg.NodeState):
            count: int = rg.var(init=0)

        def update(self, prev_state: "State") -> "State":
            return self.State(count=prev_state.count + 1)

    counter = Counter()
    system = _single_phase_system(counter)

    system.run(steps=2)
    assert system.read(counter.State.count) == 2


def test_update_can_receive_parameter_inputs_and_previous_state() -> None:
    class Accumulator(rg.Node):
        class State(rg.NodeState):
            total: int = rg.var(init=0)

        def update(
            self,
            value: int = rg.src(Source.State.value),
            *,
            prevstate: State,
        ) -> State:
            return self.State(total=prevstate.total + value)

    source = Source(value=3)
    accumulator = Accumulator()
    system = _single_phase_system(source, accumulator)

    assert system.compile_report.ok
    assert system.compile_report.inputs["Accumulator.value"] == "Source.value"
    system.run(steps=2)
    assert system.read(accumulator.State.total) == 6


def test_previous_state_requires_initial_value_or_initial_state() -> None:
    class Counter(rg.Node):
        class State(rg.NodeState):
            count: int = rg.var()

        def update(self, prev_state: State) -> State:
            return self.State(count=prev_state.count + 1)

    counter = Counter()
    system = _single_phase_system(counter)

    with pytest.raises(RuntimeError, match="define var\\(init=\\.\\.\\.\\) or pass initial_state"):
        system.step()

    system.reset(initial_state={counter.State.count: 10})
    system.step()
    assert system.read(counter.State.count) == 11


def test_bare_state_annotations_declare_vars_without_initial_values() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=2)

        def update(self) -> State:
            return self.State(value=3)

    class Doubler(rg.Node):
        class Inputs(rg.NodeInputs):
            value: int = rg.src(Source.State.value)

        class State(rg.NodeState):
            doubled: int

        def update(self, inputs: Inputs) -> State:
            return self.State(doubled=inputs.value * 2)

    source = Source()
    doubler = Doubler()
    system = _single_phase_system(source, doubler)

    assert system.compile_report.ok
    assert "Doubler.doubled" in system.compile_report.state_vars_without_initial
    system.run(steps=1)
    assert system.read(doubler.State.doubled) == 6
