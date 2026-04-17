from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


State = dict[str, Any]
Step = Callable[[State], State]


class Node:
    name: str

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__

    def run(self, state: State) -> State:
        raise NotImplementedError


@dataclass(frozen=True)
class Phase:
    name: str
    nodes: tuple[Node, ...]


class ReactiveSystem:
    def __init__(
        self,
        initial_state: State | None = None,
        step: Step | None = None,
        nodes: Iterable[Node] = (),
        phases: Iterable[Phase] = (),
    ) -> None:
        self._initial_state = dict(initial_state or {})
        self._state = dict(self._initial_state)
        self._step = step
        self._nodes = tuple(nodes)
        self._phases = tuple(phases)
        self._phase_index = 0

    def reset(self) -> None:
        self._state = dict(self._initial_state)

    def step(self) -> State:
        if self._step is not None:
            self._state = self._step(dict(self._state))
            return self.snapshot()

        state = dict(self._state)
        active_nodes = self._nodes
        if self._phases:
            active_nodes = self._phases[self._phase_index].nodes
        for node in active_nodes:
            state.update(node.run(dict(state)))
        self._state = state
        return self.snapshot()

    def run(self, steps: int = 1) -> list[State]:
        snapshots: list[State] = []
        for _ in range(steps):
            snapshots.append(self.step())
        return snapshots

    def snapshot(self) -> State:
        return dict(self._state)
