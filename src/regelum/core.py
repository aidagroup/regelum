from __future__ import annotations

from collections.abc import Callable
from typing import Any


State = dict[str, Any]
Step = Callable[[State], State]


class ReactiveSystem:
    def __init__(self, initial_state: State, step: Step) -> None:
        self._initial_state = dict(initial_state)
        self._state = dict(initial_state)
        self._step = step

    def reset(self) -> None:
        self._state = dict(self._initial_state)

    def run(self, steps: int = 1) -> None:
        for _ in range(steps):
            self._state = self._step(dict(self._state))

    def snapshot(self) -> State:
        return dict(self._state)
