from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import z3


State = dict[str, Any]
Step = Callable[[State], State]
SourceRef = str
Predicate = Callable[[State], bool]


def _normalize_source_path(path: str) -> str:
    return path.replace("::", ".").replace("/", ".").strip(". ")


class Expr:
    def evaluate(self, state: State) -> Any:
        raise NotImplementedError

    def to_z3(self, ctx: Z3Context) -> Any:
        raise NotImplementedError

    def __and__(self, other: Any) -> Expr:
        return BinaryExpr("and", self, _as_expr(other))

    def __or__(self, other: Any) -> Expr:
        return BinaryExpr("or", self, _as_expr(other))

    def __invert__(self) -> Expr:
        return UnaryExpr("not", self)

    def __eq__(self, other: object) -> Expr:  # type: ignore[override]
        return BinaryExpr("eq", self, _as_expr(other))

    def __ne__(self, other: object) -> Expr:  # type: ignore[override]
        return BinaryExpr("ne", self, _as_expr(other))


@dataclass(frozen=True)
class ConstExpr(Expr):
    value: Any

    def evaluate(self, state: State) -> Any:
        return self.value

    def to_z3(self, ctx: Z3Context) -> Any:
        return self.value


@dataclass(frozen=True)
class VarExpr(Expr):
    path: SourceRef

    def evaluate(self, state: State) -> Any:
        return state[_normalize_source_path(self.path)]

    def to_z3(self, ctx: Z3Context) -> Any:
        return ctx.variable(_normalize_source_path(self.path))


def V(path: SourceRef) -> Expr:
    return VarExpr(_normalize_source_path(path))


@dataclass(frozen=True)
class UnaryExpr(Expr):
    op: str
    operand: Expr

    def evaluate(self, state: State) -> Any:
        if self.op == "not":
            return not bool(self.operand.evaluate(state))
        raise ValueError(f"Unknown unary operator: {self.op}")

    def to_z3(self, ctx: Z3Context) -> Any:
        if self.op == "not":
            return z3.Not(self.operand.to_z3(ctx))
        raise ValueError(f"Unknown unary operator: {self.op}")


@dataclass(frozen=True)
class BinaryExpr(Expr):
    op: str
    left: Expr
    right: Expr

    def evaluate(self, state: State) -> Any:
        left = self.left.evaluate(state)
        right = self.right.evaluate(state)
        if self.op == "and":
            return bool(left) and bool(right)
        if self.op == "or":
            return bool(left) or bool(right)
        if self.op == "eq":
            return left == right
        if self.op == "ne":
            return left != right
        raise ValueError(f"Unknown binary operator: {self.op}")

    def to_z3(self, ctx: Z3Context) -> Any:
        left = self.left.to_z3(ctx)
        right = self.right.to_z3(ctx)
        if self.op == "and":
            return z3.And(left, right)
        if self.op == "or":
            return z3.Or(left, right)
        if self.op == "eq":
            return left == right
        if self.op == "ne":
            return left != right
        raise ValueError(f"Unknown binary operator: {self.op}")


def _as_expr(value: Any) -> Expr:
    if isinstance(value, Expr):
        return value
    return ConstExpr(value)


class Z3Context:
    def __init__(self, bindings: dict[str, Any] | None = None) -> None:
        self.bindings = dict(bindings or {})
        self.variables: dict[str, Any] = {}

    def variable(self, path: str) -> Any:
        if path in self.bindings:
            return self.bindings[path]
        if path not in self.variables:
            self.variables[path] = z3.Real(path)
        return self.variables[path]


Guard = Predicate | Expr


class InputValues:
    def __init__(self, values: State | None = None, **kwargs: Any) -> None:
        for key, value in dict(values or {}) | kwargs.items():
            setattr(self, key, value)

    def as_dict(self) -> State:
        return dict(vars(self))


class OutputValues(InputValues):
    pass


class NodeInputs:
    pass


class NodeOutputs:
    pass


def _namespace_annotations(namespace: type[Any]) -> dict[str, Any]:
    return dict(getattr(namespace, "__annotations__", {}))


@dataclass(frozen=True)
class OutputPort:
    name: str | None = None


@dataclass(frozen=True)
class InputPort:
    name: str | None = None
    source: SourceRef | None = None


@dataclass(frozen=True)
class BoundOutputPort:
    node: Node
    port_name: str


@dataclass(frozen=True)
class BoundInputPort:
    node: Node
    port_name: str


@dataclass(frozen=True)
class Connection:
    source: BoundOutputPort
    target: BoundInputPort


def port(reference: tuple[Node, str]) -> BoundOutputPort | BoundInputPort:
    node, name = reference
    if name in node.output_ports or name in node.outputs:
        return BoundOutputPort(node=node, port_name=name)
    return BoundInputPort(node=node, port_name=name)


def connect(
    source: BoundOutputPort | tuple[Node, str],
    target: BoundInputPort | tuple[Node, str],
) -> Connection:
    bound_source = source if isinstance(source, BoundOutputPort) else port(source)
    bound_target = target if isinstance(target, BoundInputPort) else port(target)
    if not isinstance(bound_source, BoundOutputPort):
        raise TypeError("connect() source must be an output port.")
    if not isinstance(bound_target, BoundInputPort):
        raise TypeError("connect() target must be an input port.")
    return Connection(source=bound_source, target=bound_target)


def Output(name: str | None = None) -> OutputPort:
    return OutputPort(name=name)


def Input(source: SourceRef | None = None, name: str | None = None) -> InputPort:
    return InputPort(name=name, source=source)


class Node:
    name: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    input_sources: dict[str, SourceRef] = {}
    input_ports: dict[str, InputPort] = {}
    output_ports: dict[str, OutputPort] = {}

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        input_namespace = getattr(cls, "Inputs", None)
        output_namespace = getattr(cls, "Outputs", None)

        declared_inputs: dict[str, InputPort] = {}
        if isinstance(input_namespace, type):
            for name in _namespace_annotations(input_namespace):
                value = getattr(input_namespace, name, Input())
                if not isinstance(value, InputPort):
                    value = Input()
                declared_inputs[name] = InputPort(name=name, source=value.source or name)

        declared_outputs: dict[str, OutputPort] = {}
        if isinstance(output_namespace, type):
            for name in _namespace_annotations(output_namespace):
                value = getattr(output_namespace, name, Output())
                if not isinstance(value, OutputPort):
                    value = Output()
                declared_outputs[name] = OutputPort(name=name)

        if declared_inputs:
            cls.input_ports = declared_inputs
            cls.inputs = tuple(declared_inputs)
            cls.input_sources = {
                name: port.source or name for name, port in declared_inputs.items()
            }

        if declared_outputs:
            cls.output_ports = declared_outputs
            cls.outputs = tuple(declared_outputs)

    def __init__(self, name: str | None = None) -> None:
        self.name = name or self.__class__.__name__

    def read_inputs(self, state: State) -> InputValues:
        if self.input_ports:
            return InputValues(
                {
                    name: state[_normalize_source_path(port.source or name)]
                    for name, port in self.input_ports.items()
                }
            )
        if not self.inputs:
            return InputValues(state)
        values: State = {}
        for name in self.inputs:
            source = self.input_sources.get(name, name)
            values[name] = state[_normalize_source_path(source)]
        return InputValues(values)

    def write_outputs(self, values: State) -> State:
        if isinstance(values, OutputValues):
            values = values.as_dict()
        if self.output_ports:
            return {
                name: values[port.name]
                for name, port in self.output_ports.items()
            }
        if not self.outputs:
            return dict(values)
        return {name: values[name] for name in self.outputs}

    def run(self, state: InputValues) -> State | OutputValues:
        raise NotImplementedError


@dataclass(frozen=True)
class Phase:
    name: str
    nodes: tuple[Node, ...]
    transitions: tuple[Transition, ...] = ()
    is_initial: bool = False


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
        self._phase_index = next(
            (index for index, phase in enumerate(self._phases) if phase.is_initial),
            0,
        )

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
            values = node.run(node.read_inputs(state))
            state.update(node.write_outputs(values))
        self._state = state
        self._advance_phase()
        return self.snapshot()

    def run(self, steps: int = 1) -> list[State]:
        snapshots: list[State] = []
        for _ in range(steps):
            snapshots.append(self.step())
        return snapshots

    def snapshot(self) -> State:
        return dict(self._state)

    def _advance_phase(self) -> None:
        if not self._phases:
            return
        phase = self._phases[self._phase_index]
        for transition in phase.transitions:
            if transition.enabled(self._state):
                if transition.target is terminate:
                    return
                self._phase_index = self._phase_names()[transition.target]
                return

    def _phase_names(self) -> dict[str, int]:
        return {phase.name: index for index, phase in enumerate(self._phases)}


@dataclass(frozen=True)
class Transition:
    target: str | TerminateTarget
    guard: Guard

    def enabled(self, state: State) -> bool:
        if isinstance(self.guard, Expr):
            return bool(self.guard.evaluate(state))
        return bool(self.guard(state))


class TerminateTarget:
    pass


terminate = TerminateTarget()


def always(_: State) -> bool:
    return True


def when(guard: Guard, target: str | TerminateTarget) -> Transition:
    return Transition(target=target, guard=guard)


def otherwise(target: str | TerminateTarget) -> Transition:
    return Transition(target=target, guard=always)


def If(guard: Guard, target: str | TerminateTarget) -> Transition:
    return when(guard, target)


def ElseIf(guard: Guard, target: str | TerminateTarget) -> Transition:
    return when(guard, target)


def Elif(guard: Guard, target: str | TerminateTarget) -> Transition:
    return ElseIf(guard, target)


def Else(target: str | TerminateTarget) -> Transition:
    return otherwise(target)


def Goto(target: str | TerminateTarget) -> Transition:
    return otherwise(target)
