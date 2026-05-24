import math
from typing import Any, cast

import casadi as ca
import numpy as np

import regelum as rg


class SwitchNode(rg.ODENode):
    class Inputs(rg.NodeInputs):
        time: float = rg.src(rg.Clock.time)

    class State(rg.NodeState):
        x: float = rg.var(init=0.0)

    def dstate(self, inputs: Inputs, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=ca.if_else(inputs.time < 0.5, 1.0, -1.0))


class TrigNode(rg.ODENode):
    class State(rg.NodeState):
        x: float = rg.var(init=1.0)

    def dstate(self, inputs: rg.NodeInputs, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=-ca.sin(state.x))


class BadMathNode(rg.ODENode):
    class State(rg.NodeState):
        x: float = rg.var(init=1.0)

    def dstate(self, inputs: rg.NodeInputs, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=-math.sin(state.x))


class InputsStateTimeNode(rg.ODENode):
    class Inputs(rg.NodeInputs):
        u: float = rg.src("u.value")

    class State(rg.NodeState):
        x: float = rg.var(init=0.0)

    def dstate(  # ty: ignore[invalid-method-override]
        self,
        inputs: Inputs,
        state: State,
        time: Any,
    ) -> State:
        return self.State(x=inputs.u + time)


class ReorderedInputsStateTimeNode(rg.ODENode):
    class Inputs(rg.NodeInputs):
        u: float = rg.src("u.value")

    class State(rg.NodeState):
        x: float = rg.var(init=0.0)

    def dstate(  # ty: ignore[invalid-method-override]
        self,
        time: Any,
        state: State,
        inputs: Inputs,
    ) -> State:
        return self.State(x=state.x * 0.0 + inputs.u + time)


class TypedOnlyInputsStateNode(rg.ODENode):
    class Inputs(rg.NodeInputs):
        u: float = rg.src("u.value")

    class State(rg.NodeState):
        x: float = rg.var(init=0.0)

    def dstate(  # ty: ignore[invalid-method-override]
        self,
        control: Inputs,
        memory: State,
    ) -> State:
        return self.State(x=memory.x * 0.0 + control.u)


class DirectDstateInputsNode(rg.ODENode):
    class State(rg.NodeState):
        x: float = rg.var(init=0.0)

    def dstate(  # ty: ignore[invalid-method-override]
        self,
        time: Any,
        state: State,
        a: float = rg.src("a.value"),
        b: float = rg.src("b.value"),
    ) -> State:
        return self.State(x=state.x * 0.0 + a + 2.0 * b + time)


class DirectLazyDstateInputsNode(rg.ODENode):
    class State(rg.NodeState):
        x: float = rg.var(init=0.0)

    def dstate(  # ty: ignore[invalid-method-override]
        self,
        state: State,
        u: float = rg.src(lambda: LazyDstateInputSource.State.u),
    ) -> State:
        return self.State(x=state.x * 0.0 + u)


class LazyDstateInputSource(rg.ODENode):
    class State(rg.NodeState):
        u: float = rg.var(init=4.0)

    def dstate(self, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(u=0.0 * state.u)


class InputsTimeNode(rg.ODENode):
    class Inputs(rg.NodeInputs):
        u: float = rg.src("u.value")

    class State(rg.NodeState):
        x: float = rg.var(init=0.0)

    def dstate(self, inputs: Inputs, time: Any) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=inputs.u + time)


class TimeNode(rg.ODENode):
    class State(rg.NodeState):
        x: float = rg.var(init=0.0)

    def dstate(self, time: Any) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=time)


class InputsNode(rg.ODENode):
    class Inputs(rg.NodeInputs):
        u: float = rg.src("u.value")

    class State(rg.NodeState):
        x: float = rg.var(init=0.0)

    def dstate(self, inputs: Inputs) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=inputs.u)


class StateNode(rg.ODENode):
    class State(rg.NodeState):
        x: float = rg.var(init=1.0)

    def dstate(self, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=state.x)


class NumpyVectorNode(rg.ODENode):
    class Inputs(rg.NodeInputs):
        u: np.ndarray = rg.src("u.value")

    class State(rg.NodeState):
        x: np.ndarray = rg.var(init=lambda: np.zeros(3))

    def dstate(self, inputs: Inputs, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=state.x + inputs.u)


class MatrixNode(rg.ODENode):
    class State(rg.NodeState):
        x: np.ndarray = rg.var(init=lambda: np.zeros((2, 2)))

    def dstate(self, state: State) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=state.x * 0.0 + 1.0)


class ListTupleVectorNode(rg.ODENode):
    class State(rg.NodeState):
        x: list[float] = rg.var(init=lambda: [0.0, 0.0])
        y: tuple[float, float] = rg.var(init=(1.0, 2.0))

    def dstate(self, state: State) -> State:  # ty: ignore[invalid-method-override]
        x = cast(Any, state.x)
        y = cast(Any, state.y)
        return self.State(x=x + y, y=-y)


class AbsoluteTimeNode(rg.ODENode):
    class State(rg.NodeState):
        x: float = rg.var(init=0.0)

    def dstate(self, time: Any) -> State:  # ty: ignore[invalid-method-override]
        return self.State(x=time)
