from __future__ import annotations

from fractions import Fraction
from typing import Any

import regelum as rg


def _one_phase_system(
    *nodes: rg.Node,
    base_dt: Fraction | int | str = "auto",
) -> rg.PhasedReactiveSystem:
    return rg.PhasedReactiveSystem(
        phases=[
            rg.Phase(
                "tick",
                nodes=nodes,
                transitions=(rg.Goto(rg.terminate),),
                is_initial=True,
            )
        ],
        base_dt=base_dt,
    )


class Counter(rg.Node):
    class Inputs(rg.NodeInputs):
        previous: int = rg.src(lambda: Counter.State.value)

    class State(rg.NodeState):
        value: int = rg.var(init=0)

    def update(self, inputs: Inputs) -> State:
        return self.State(value=inputs.previous + 1)


class ScheduledCounter(rg.Node):
    def __init__(self, *, name: str | None = None, dt: Fraction | int | str | None = None) -> None:
        super().__init__(name=name, dt=dt)
        self.count = 0

    class State(rg.NodeState):
        value: int = rg.var(init=0)

    def update(self) -> State:
        self.count += 1
        return self.State(value=self.count)


class ClockLog(rg.Node):
    class Inputs(rg.NodeInputs):
        previous_ticks: tuple[int, ...] = rg.src(lambda: ClockLog.State.ticks)
        previous_times: tuple[float, ...] = rg.src(lambda: ClockLog.State.times)
        tick: int = rg.src(rg.Clock.tick)
        time: float = rg.src(rg.Clock.time)

    class State(rg.NodeState):
        ticks: tuple[int, ...] = rg.var(init=())
        times: tuple[float, ...] = rg.var(init=())

    def update(self, inputs: Inputs) -> State:
        return self.State(
            ticks=(*inputs.previous_ticks, inputs.tick),
            times=(*inputs.previous_times, inputs.time),
        )


class TimeDrivenIntegrator(rg.ODENode):
    class State(rg.NodeState):
        x: float = rg.var(init=0.0)

    def dstate(  # ty: ignore[invalid-method-override]
        self,
        inputs: rg.NodeInputs,
        state: State,
        *,
        time: Any,
    ) -> State:
        return self.State(x=time)


class ConstantIntegrator(rg.ODENode):
    class State(rg.NodeState):
        x: float = rg.var(init=0.0)

    def dstate(self, inputs: rg.NodeInputs, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=1.0)


class CoupledSourceIntegrator(rg.ODENode):
    class State(rg.NodeState):
        x: float = rg.var(init=1.0)

    def dstate(self, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=0.0 * state.x)


class CoupledSinkIntegrator(rg.ODENode):
    class Inputs(rg.NodeInputs):
        source_x: float = rg.src(lambda: CoupledSourceIntegrator.State.x)

    class State(rg.NodeState):
        x: float = rg.var(init=0.0)

    def dstate(self, inputs: Inputs) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=inputs.source_x)
