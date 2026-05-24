from typing import cast

import pytest

import regelum as rg
from tests.core.system._support import _tick_system


def test_var_initial_accepts_zero_argument_callable() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=lambda: 7)

    system = _tick_system([Source()])

    assert system.snapshot()["Source.value"] == 7


def test_var_initial_accepts_node_argument_callable() -> None:
    class Source(rg.Node):
        def __init__(self, value: int, *, name: str | None = None) -> None:
            super().__init__(name=name)
            self.value = value

        class State(rg.NodeState):
            value: int = rg.var(init=lambda self: cast(Source, self).value)

    left = Source(3, name="left")
    right = Source(7, name="right")

    system = _tick_system([left, right])

    assert system.snapshot() == {
        "left.value": 3,
        "right.value": 7,
    }


def test_var_initial_rejects_callable_with_too_many_required_arguments() -> None:
    class Source(rg.Node):
        class State(rg.NodeState):
            value: int = rg.var(init=lambda first, second: 1)

    with pytest.raises(rg.CompileError) as exc_info:
        _tick_system([Source()])

    assert any(
        issue.location == "Source.value"
        and "var init callable must accept zero arguments or one node argument" in issue.message
        for issue in exc_info.value.report.issues
    )
