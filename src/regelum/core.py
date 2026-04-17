from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


State = dict[str, Any]
Step = Callable[[State], State]


class Node:
    name: str

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__

    def run(self, state: State) -> State:
        raise NotImplementedError


class ReactiveSystem:
    def __init__(
        self,
        initial_state: State | None = None,
        step: Step | None = None,
        nodes: Iterable[Node] = (),
    ) -> None:
        self._initial_state = dict(initial_state or {})
        self._state = dict(self._initial_state)
        self._step = step
        self._nodes = tuple(nodes)

    def reset(self) -> None:
        self._state = dict(self._initial_state)

    def step(self) -> None:
        if self._step is not None:
            self._state = self._step(dict(self._state))
            return

        state = dict(self._state)
        for node in self._nodes:
            state.update(node.run(dict(state)))
        self._state = state

    def run(self, steps: int = 1) -> None:
        for _ in range(steps):
            self.step()

    def snapshot(self) -> State:
        return dict(self._state)
