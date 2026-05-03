from __future__ import annotations

import inspect
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from functools import reduce
from itertools import product
from math import gcd
from typing import (
    Annotated,
    Any,
    Generic,
    Literal,
    Protocol,
    TypeGuard,
    TypeVar,
    cast,
    dataclass_transform,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

import z3

T = TypeVar("T")
StateSnapshot = dict[str, Any]
Predicate = Callable[[StateSnapshot], bool]
InitialValue = Any | Callable[[], Any] | Callable[[Any], Any]
_MISSING = object()
TimeStep = Fraction | int | str
BaseTimeStep = TimeStep | Literal["auto"]
SYSTEM_CLOCK = "Clock"
SYSTEM_OUTPUTS = frozenset((f"{SYSTEM_CLOCK}.tick", f"{SYSTEM_CLOCK}.time"))


class ConnectablePort(Protocol):
    def connect(self, other: Any) -> Connection: ...


@dataclass(frozen=True)
class SystemSource:
    path: str


class _ClockNamespace:
    tick = SystemSource(f"{SYSTEM_CLOCK}.tick")
    time = SystemSource(f"{SYSTEM_CLOCK}.time")


Clock = _ClockNamespace()


class Expr:
    @property
    def variables(self) -> frozenset[str]:
        raise NotImplementedError

    def evaluate(self, state: StateSnapshot) -> bool:
        value = self._eval(state)
        if not isinstance(value, bool):
            raise TypeError(f"Guard expression must evaluate to bool, got {value!r}.")
        return value

    def to_z3(self, ctx: Z3Context) -> z3.BoolRef:
        value = self._to_z3(ctx)
        if z3.is_bool(value):
            return cast(z3.BoolRef, value)
        raise TypeError("Guard expression must compile to a z3 Bool expression.")

    def _eval(self, state: StateSnapshot) -> Any:
        raise NotImplementedError

    def _to_z3(self, ctx: Z3Context) -> Any:
        raise NotImplementedError

    def __and__(self, other: Any) -> Expr:
        return BinaryExpr("and", self, _as_expr(other))

    def __rand__(self, other: Any) -> Expr:
        return BinaryExpr("and", _as_expr(other), self)

    def __or__(self, other: Any) -> Expr:
        return BinaryExpr("or", self, _as_expr(other))

    def __ror__(self, other: Any) -> Expr:
        return BinaryExpr("or", _as_expr(other), self)

    def __invert__(self) -> Expr:
        return UnaryExpr("not", self)

    def __eq__(self, other: object) -> Expr:  # type: ignore[override]  # ty: ignore[invalid-method-override]
        return BinaryExpr("eq", self, _as_expr(other))

    def __ne__(self, other: object) -> Expr:  # type: ignore[override]  # ty: ignore[invalid-method-override]
        return BinaryExpr("ne", self, _as_expr(other))

    def __lt__(self, other: Any) -> Expr:
        return BinaryExpr("lt", self, _as_expr(other))

    def __le__(self, other: Any) -> Expr:
        return BinaryExpr("le", self, _as_expr(other))

    def __gt__(self, other: Any) -> Expr:
        return BinaryExpr("gt", self, _as_expr(other))

    def __ge__(self, other: Any) -> Expr:
        return BinaryExpr("ge", self, _as_expr(other))

    def __bool__(self) -> bool:
        raise TypeError("Use '&', '|', and '~' to compose regelum guard expressions.")


@dataclass(frozen=True, eq=False)
class ConstExpr(Expr):
    value: Any

    @property
    def variables(self) -> frozenset[str]:
        return frozenset()

    def _eval(self, state: StateSnapshot) -> Any:
        return self.value

    def _to_z3(self, ctx: Z3Context) -> Any:
        return _z3_value(self.value)


@dataclass(frozen=True, eq=False)
class VarExpr(Expr):
    path: str
    output_source: OutputSource[Any] | None = None

    @property
    def variables(self) -> frozenset[str]:
        return frozenset((self.path,))

    def _eval(self, state: StateSnapshot) -> Any:
        return state[self.path]

    def _to_z3(self, ctx: Z3Context) -> Any:
        return ctx.variable(self.path)


@dataclass(frozen=True, eq=False)
class UnaryExpr(Expr):
    op: Literal["not"]
    operand: Expr

    @property
    def variables(self) -> frozenset[str]:
        return self.operand.variables

    def _eval(self, state: StateSnapshot) -> Any:
        if self.op == "not":
            return not self.operand.evaluate(state)
        raise ValueError(f"Unknown unary operation {self.op!r}.")

    def _to_z3(self, ctx: Z3Context) -> Any:
        if self.op == "not":
            return z3.Not(self.operand.to_z3(ctx))
        raise ValueError(f"Unknown unary operation {self.op!r}.")


@dataclass(frozen=True, eq=False)
class BinaryExpr(Expr):
    op: Literal["and", "or", "eq", "ne", "lt", "le", "gt", "ge"]
    left: Expr
    right: Expr

    @property
    def variables(self) -> frozenset[str]:
        return self.left.variables | self.right.variables

    def _eval(self, state: StateSnapshot) -> Any:
        if self.op == "and":
            return self.left.evaluate(state) and self.right.evaluate(state)
        if self.op == "or":
            return self.left.evaluate(state) or self.right.evaluate(state)
        left = self.left._eval(state)
        right = self.right._eval(state)
        if self.op == "eq":
            return left == right
        if self.op == "ne":
            return left != right
        if self.op == "lt":
            return left < right
        if self.op == "le":
            return left <= right
        if self.op == "gt":
            return left > right
        if self.op == "ge":
            return left >= right
        raise ValueError(f"Unknown binary operation {self.op!r}.")

    def _to_z3(self, ctx: Z3Context) -> Any:
        if self.op == "and":
            return z3.And(self.left.to_z3(ctx), self.right.to_z3(ctx))
        if self.op == "or":
            return z3.Or(self.left.to_z3(ctx), self.right.to_z3(ctx))
        left = self.left._to_z3(ctx)
        right = self.right._to_z3(ctx)
        if self.op == "eq":
            return left == right
        if self.op == "ne":
            return left != right
        if self.op == "lt":
            return left < right
        if self.op == "le":
            return left <= right
        if self.op == "gt":
            return left > right
        if self.op == "ge":
            return left >= right
        raise ValueError(f"Unknown binary operation {self.op!r}.")


class Z3Context:
    def __init__(
        self,
        output_types: dict[str, type[Any]],
        domains: dict[str, tuple[Any, ...]] | None = None,
        bindings: dict[str, Any] | None = None,
    ) -> None:
        self.output_types = output_types
        self.domains = domains or {}
        self.bindings = bindings or {}
        self.variables: dict[str, Any] = {}

    def variable(self, path: str) -> Any:
        path = _normalize_output_path(path)
        if path in self.bindings:
            return self.bindings[path]
        if path in self.variables:
            return self.variables[path]
        output_type = self.output_types.get(path)
        if output_type is bool:
            variable: Any = z3.Bool(path)
        elif output_type is int:
            variable = z3.Int(path)
        elif output_type is float:
            variable = z3.Real(path)
        elif _is_enum_type(output_type):
            variable = z3.Int(path)
        else:
            raise TypeError(f"Output {path!r} has no z3-supported type.")
        self.variables[path] = variable
        return variable

    def domain_constraints(self, paths: Iterable[str]) -> list[z3.BoolRef]:
        constraints: list[z3.BoolRef] = []
        for path in sorted({_normalize_output_path(path) for path in paths}):
            domain = self.domains.get(path)
            if domain is None:
                continue
            variable = self.variable(path)
            constraints.append(
                cast(z3.BoolRef, z3.Or(*[variable == _z3_value(value) for value in domain]))
            )
        return constraints


def _as_expr(value: Any) -> Expr:
    if isinstance(value, Expr):
        return value
    return ConstExpr(value)


def V(output: Any) -> Expr:
    return VarExpr(_source_path(output), output_source=output)


type Guard = Predicate | Expr


class OutputPort(Generic[T]):
    def __init__(
        self,
        initial: InitialValue = _MISSING,
        domain: Iterable[T] | None = None,
    ) -> None:
        self.initial = initial
        self.domain = tuple(domain) if domain is not None else None
        self.name: str | None = None
        self.node_cls: type[Node] | None = None

    def __set_name__(self, owner: type[NodeOutputs], name: str) -> None:
        self.name = name

    @overload
    def __get__(self, instance: None, owner: type[NodeOutputs]) -> OutputPort[T]: ...

    @overload
    def __get__(self, instance: NodeOutputs, owner: type[NodeOutputs]) -> T: ...

    def __get__(
        self,
        instance: NodeOutputs | None,
        owner: type[NodeOutputs],
    ) -> OutputPort[T] | T:
        if instance is None:
            return self
        if self.name is None:
            raise AttributeError("Output is not bound to an attribute name.")
        return instance.__dict__[self.name]

    def initial_value(self, node: Node) -> Any:
        if self.initial is _MISSING:
            raise RuntimeError(f"Output {self.path} does not define an initial value.")
        if callable(self.initial):
            return _call_initial_value(self.initial, node)
        return self.initial

    def finite_domain(self, node: Node) -> tuple[Any, ...] | None:
        if self.initial is _MISSING:
            return None
        if self.domain is not None:
            return self.domain
        try:
            initial = self.initial_value(node)
        except Exception:
            return None
        if isinstance(initial, bool):
            return (False, True)
        return None

    @property
    def path(self) -> str:
        if self.node_cls is None or self.name is None:
            return "<unbound>"
        return f"{self.node_cls.__name__}.{self.name}"

    def __repr__(self) -> str:
        return self.path


type ResolvedOutputSource[T] = OutputPort[T] | BoundOutputPort[T] | SystemSource | str
type OutputSource[T] = ResolvedOutputSource[T] | Callable[[], ResolvedOutputSource[T]]


@dataclass(frozen=True)
class BoundOutputPort(Generic[T]):
    node: Node
    port: Any

    @property
    def path(self) -> str:
        if self.port.name is None:
            return "<unbound>"
        return f"{self.node.node_id}.{self.port.name}"

    def connect(self, input_port: Any) -> Connection:
        connection = connect(input_port, self)
        connection.input.node._connections[connection.input.path] = connection
        return connection


@dataclass(frozen=True)
class BoundInputPort(Generic[T]):
    node: Node
    port: Any

    @property
    def path(self) -> str:
        if self.port.name is None:
            return "<unbound>"
        return f"{self.node.node_id}.{self.port.name}"

    def connect(self, output: Any) -> Connection:
        connection = connect(self, output)
        self.node._connections[self.path] = connection
        return connection


@dataclass(frozen=True)
class Connection:
    input: BoundInputPort[Any]
    source: Any


def connect(
    input_port: Any,
    output: Any,
) -> Connection:
    if not isinstance(input_port, BoundInputPort):
        raise TypeError(f"connect(...) expects an input port on the left side; got {input_port!r}.")
    if not isinstance(output, (BoundOutputPort, OutputPort, SystemSource, str)):
        raise TypeError(
            "connect(...) expects an output port or output reference, or system source on "
            "the right side; "
            f"got {output!r}."
        )
    return Connection(input=input_port, source=output)


def port(reference: Any) -> ConnectablePort:
    return cast(ConnectablePort, reference)


class InputPort(Generic[T]):
    def __init__(
        self,
        source: OutputSource[T] | None = None,
        default: T | None = None,
    ) -> None:
        self.source = source
        self.default = default
        self.name: str | None = None
        self.node_cls: type[Node] | None = None

    def __set_name__(self, owner: type[NodeInputs], name: str) -> None:
        self.name = name

    @overload
    def __get__(self, instance: None, owner: type[NodeInputs]) -> InputPort[T]: ...

    @overload
    def __get__(self, instance: NodeInputs, owner: type[NodeInputs]) -> T: ...

    def __get__(
        self,
        instance: NodeInputs | None,
        owner: type[NodeInputs],
    ) -> InputPort[T] | T:
        if instance is None:
            return self
        if self.name is None:
            raise AttributeError("Input is not bound to an attribute name.")
        return instance.__dict__.get(self.name, self.default)

    @property
    def path(self) -> str:
        if self.node_cls is None or self.name is None:
            return "<unbound>"
        return f"{self.node_cls.__name__}.{self.name}"

    def __repr__(self) -> str:
        return self.path


def Output(
    *,
    initial: InitialValue = _MISSING,
    domain: Iterable[Any] | None = None,
) -> Any:
    return OutputPort(initial=initial, domain=domain)


def Input(
    source: Any = None,
    *,
    default: Any = None,
) -> Any:
    return InputPort(source=source, default=default)


def _call_initial_value(initial: Callable[..., Any], node: Node) -> Any:
    try:
        signature = inspect.signature(initial)
    except (TypeError, ValueError):
        return initial()

    required_positionals = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    if len(required_positionals) == 0:
        return initial()
    if len(required_positionals) == 1:
        return initial(node)
    raise TypeError("Output initial callable must accept zero arguments or one node argument.")


def _accepts_keyword(callable_: Callable[..., Any], keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == keyword and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def _parse_time_step(value: Any, *, field_name: str) -> Fraction:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a positive Fraction, int, or decimal string.")
    if isinstance(value, Fraction):
        parsed = value
    elif isinstance(value, int):
        parsed = Fraction(value, 1)
    elif isinstance(value, str):
        parsed = Fraction(value)
    elif isinstance(value, float):
        raise TypeError(
            f"{field_name} must not be a float; use a decimal string or Fraction instead."
        )
    else:
        raise TypeError(f"{field_name} must be a positive Fraction, int, or decimal string.")
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive.")
    return parsed


def _gcd_fraction(values: Iterable[Fraction]) -> Fraction:
    value_tuple = tuple(values)
    if not value_tuple:
        return Fraction(1, 1)
    common_denominator = reduce(_lcm, (value.denominator for value in value_tuple), 1)
    scaled = [value.numerator * (common_denominator // value.denominator) for value in value_tuple]
    return Fraction(reduce(gcd, scaled), common_denominator)


def _lcm(left: int, right: int) -> int:
    return abs(left * right) // gcd(left, right)


@dataclass_transform(field_specifiers=(Input,))
class NodeInputs:
    def __init__(self, **values: Any) -> None:
        for name, value in values.items():
            setattr(self, name, value)


@dataclass_transform(field_specifiers=(Output,))
class NodeOutputs:
    def __init__(self, **values: Any) -> None:
        for name, value in values.items():
            setattr(self, name, value)


class _BoundPortNamespace:
    def __init__(
        self,
        node: Node,
        nested_cls: type[NodeInputs] | type[NodeOutputs],
        ports: dict[str, InputPort[Any]] | dict[str, OutputPort[Any]],
        bound_type: type[BoundInputPort[Any]] | type[BoundOutputPort[Any]],
    ) -> None:
        self._node = node
        self._nested_cls = nested_cls
        self._ports = ports
        self._bound_type = bound_type

    def __getattr__(self, name: str) -> Any:
        try:
            port = self._ports[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return self._bound_type(self._node, port)

    def __call__(self, **values: Any) -> Any:
        return self._nested_cls(**values)


class Node:
    Inputs = NodeInputs
    Outputs = NodeOutputs

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        input_namespace_name, input_namespace_cls = _find_port_namespace(
            cls,
            NodeInputs,
            "input",
        )
        output_namespace_name, output_namespace_cls = _find_port_namespace(
            cls,
            NodeOutputs,
            "output",
        )
        if input_namespace_name is not None:
            _install_annotated_ports(cls, input_namespace_cls, InputPort)
        if output_namespace_name is not None:
            _install_annotated_ports(cls, output_namespace_cls, OutputPort)
        nested_inputs = _collect_ports(input_namespace_cls, InputPort)
        run_inputs = _collect_run_input_ports(cls)
        cls._input_declaration_error: str | None = None
        cls._input_namespace_name = input_namespace_name
        cls._input_namespace_cls = input_namespace_cls
        cls._output_namespace_name = output_namespace_name
        cls._output_namespace_cls = output_namespace_cls
        if nested_inputs and run_inputs:
            cls._input_declaration_error = (
                "define inputs either as a NodeInputs namespace or as run(...) parameters, not both"
            )
            cls._run_input_mode = "object"
            cls._inputs = nested_inputs
        elif run_inputs:
            inputs_cls = type("Inputs", (NodeInputs,), dict(run_inputs))
            for name, port in run_inputs.items():
                port.__set_name__(inputs_cls, name)
            cls.Inputs = inputs_cls  # ty: ignore[invalid-assignment]
            cls._input_namespace_name = "Inputs"
            cls._input_namespace_cls = inputs_cls
            cls._run_input_mode = "parameters"
            cls._inputs = run_inputs
        else:
            cls._run_input_mode = "object" if nested_inputs else "none"
            cls._inputs = nested_inputs
        cls._outputs = _collect_ports(output_namespace_cls, OutputPort)
        for port in cls._inputs.values():
            port.node_cls = cls
        for port in cls._outputs.values():
            port.node_cls = cls
        original_init = cls.__dict__.get("__init__")

        def __init__(
            self: Node,
            *args: Any,
            name: str | None = None,
            dt: TimeStep | None = None,
            **kwargs: Any,
        ) -> None:
            self._initialize_node(name=name, dt=dt)
            if original_init is not None:
                if _accepts_keyword(original_init, "name"):
                    if _accepts_keyword(original_init, "dt") and dt is not None:
                        original_init(self, *args, name=name, dt=dt, **kwargs)
                    else:
                        original_init(self, *args, name=name, **kwargs)
                else:
                    if _accepts_keyword(original_init, "dt") and dt is not None:
                        original_init(self, *args, dt=dt, **kwargs)
                    else:
                        original_init(self, *args, **kwargs)
                if name is not None or dt is not None:
                    self._initialize_node(name=name, dt=dt)
            self._bind_ports()

        cls.__init__ = __init__  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]

    def __init__(self, *, name: str | None = None, dt: TimeStep | None = None) -> None:
        self._initialize_node(name=name, dt=dt)
        self._bind_ports()

    def _initialize_node(
        self,
        *,
        name: str | None = None,
        dt: TimeStep | None = None,
    ) -> None:
        class_name = getattr(self.__class__, "name", None)
        self.node_id = name or class_name or self.__class__.__name__
        self.name = self.node_id
        self._name_is_explicit = name is not None
        self._schedule_dt = (
            _parse_time_step(dt, field_name=f"{self.node_id}.dt") if dt is not None else None
        )
        self._connections: dict[str, Connection] = {}

    def _bind_ports(self) -> None:
        input_namespace_name = self.__class__._input_namespace_name
        if input_namespace_name is not None:
            setattr(
                self,
                input_namespace_name,
                _BoundPortNamespace(
                    self,
                    cast(type[NodeInputs], self.__class__._input_namespace_cls),
                    self.__class__._inputs,
                    BoundInputPort,
                ),
            )
        output_namespace_name = self.__class__._output_namespace_name
        if output_namespace_name is not None:
            setattr(
                self,
                output_namespace_name,
                _BoundPortNamespace(
                    self,
                    cast(type[NodeOutputs], self.__class__._output_namespace_cls),
                    self.__class__._outputs,
                    BoundOutputPort,
                ),
            )

    def run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def _find_port_namespace(
    node_cls: type[Node],
    namespace_base: type[NodeInputs] | type[NodeOutputs],
    namespace_role: str,
) -> tuple[str | None, type[NodeInputs] | type[NodeOutputs] | None]:
    own_namespaces = [
        (name, value)
        for name, value in node_cls.__dict__.items()
        if (
            isinstance(value, type)
            and issubclass(value, namespace_base)
            and value is not namespace_base
        )
    ]
    if len(own_namespaces) > 1:
        names = ", ".join(name for name, _ in own_namespaces)
        raise TypeError(
            f"{node_cls.__name__} may define zero or one {namespace_role} namespace; "
            f"found {len(own_namespaces)}: {names}."
        )
    if own_namespaces:
        return own_namespaces[0]
    for base_cls in node_cls.__mro__[1:]:
        namespace_name = getattr(base_cls, f"_{namespace_role}_namespace_name", None)
        namespace_cls = getattr(base_cls, f"_{namespace_role}_namespace_cls", None)
        if namespace_name is not None and namespace_cls is not None:
            return namespace_name, namespace_cls
    return None, None


def _collect_ports(
    nested_cls: type[NodeInputs] | type[NodeOutputs] | None,
    port_type: type[InputPort[Any]] | type[OutputPort[Any]],
) -> dict[str, Any]:
    if nested_cls is None:
        return {}
    return {name: value for name, value in vars(nested_cls).items() if isinstance(value, port_type)}


def _collect_run_input_ports(node_cls: type[Node]) -> dict[str, InputPort[Any]]:
    run = node_cls.__dict__.get("run")
    if run is None:
        return {}
    try:
        signature = inspect.signature(run)
    except (TypeError, ValueError):
        return {}
    inputs: dict[str, InputPort[Any]] = {}
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue
        if isinstance(parameter.default, InputPort):
            inputs[name] = parameter.default
    return inputs


def _install_annotated_ports(
    node_cls: type[Node],
    nested_cls: type[NodeInputs] | type[NodeOutputs] | None,
    port_type: type[InputPort[Any]] | type[OutputPort[Any]],
) -> None:
    if nested_cls is None:
        return
    module = sys.modules[node_cls.__module__]
    try:
        hints = get_type_hints(
            nested_cls,
            globalns=vars(module),
            localns={node_cls.__name__: node_cls},
            include_extras=True,
        )
    except Exception:
        return
    for name, annotation in hints.items():
        if hasattr(nested_cls, name):
            continue
        port = _port_from_annotation(annotation, port_type)
        if port is None:
            port = port_type()
        setattr(nested_cls, name, port)
        port.__set_name__(cast(Any, nested_cls), name)


def _port_from_annotation(
    annotation: Any,
    port_type: type[InputPort[Any]] | type[OutputPort[Any]],
) -> InputPort[Any] | OutputPort[Any] | None:
    if get_origin(annotation) is not Annotated:
        return None
    for metadata in get_args(annotation)[1:]:
        if isinstance(metadata, port_type):
            return metadata
    return None


@dataclass(frozen=True)
class StepRecord:
    phase: str
    node: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]


type NodeRef = Node
type PhaseRef = str | Phase | TerminateTarget | None
type TransitionKind = Literal["if", "elseif", "else", "goto"]


@dataclass(frozen=True)
class Phase:
    name: str
    nodes: tuple[NodeRef, ...]
    transitions: tuple[Transition, ...]
    is_initial: bool = False

    def __post_init__(self) -> None:
        for node in self.nodes:
            if not isinstance(node, Node):
                got = getattr(node, "__name__", repr(node))
                raise TypeError(
                    f"Phase.nodes accepts node instances only; got {got} in phase {self.name!r}."
                )


@dataclass(frozen=True)
class Transition:
    predicate: Guard
    target: PhaseRef = None
    name: str = "transition"
    kind: TransitionKind = "if"


@dataclass(frozen=True)
class EffectiveTransition:
    predicate: Guard
    target: PhaseRef = None
    name: str = "transition"


class TerminateTarget:
    def __repr__(self) -> str:
        return "terminate"


terminate = TerminateTarget()


def always(_: StateSnapshot) -> bool:
    return True


def If(
    predicate: Guard,
    target: PhaseRef,
    *,
    name: str = "if",
) -> Transition:
    return Transition(predicate=predicate, target=target, name=name, kind="if")


def ElseIf(
    predicate: Guard,
    target: PhaseRef,
    *,
    name: str = "elseif",
) -> Transition:
    return Transition(predicate=predicate, target=target, name=name, kind="elseif")


def Elif(
    predicate: Guard,
    target: PhaseRef,
    *,
    name: str = "elseif",
) -> Transition:
    return ElseIf(predicate, target, name=name)


def Else(
    target: PhaseRef,
    *,
    name: str = "else",
) -> Transition:
    return Transition(predicate=ConstExpr(True), target=target, name=name, kind="else")


def Goto(
    target: PhaseRef,
    *,
    name: str = "goto",
) -> Transition:
    return Transition(predicate=ConstExpr(True), target=target, name=name, kind="goto")


@dataclass(frozen=True)
class CompileIssue:
    location: str
    message: str


@dataclass(frozen=True)
class CompileReport:
    nodes: tuple[str, ...]
    inputs: dict[str, str]
    outputs: tuple[str, ...]
    issues: tuple[CompileIssue, ...]
    warnings: tuple[CompileIssue, ...] = ()
    phase_schedules: dict[str, tuple[str, ...]] = field(default_factory=dict)
    phase_dependency_edges: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)
    outputs_without_initial: tuple[str, ...] = ()
    required_initial_outputs: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.issues

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)

    @property
    def unlinked_inputs(self) -> tuple[str, ...]:
        return tuple(
            issue.location
            for issue in self.issues
            if issue.message == "input source is not connected"
        )

    @property
    def linked_inputs(self) -> tuple[str, ...]:
        return tuple(sorted(self.inputs))

    @property
    def minimal_initial_outputs(self) -> tuple[str, ...]:
        return tuple(sorted(self.required_initial_outputs))

    def format(self) -> str:
        lines = [
            f"ok = {self.ok}",
            f"issues = {_format_issue_list(self.issues)}",
            f"warnings = {_format_issue_list(self.warnings)}",
            f"minimal_initial_outputs = {self.minimal_initial_outputs}",
            f"outputs_without_initial = {self.outputs_without_initial}",
            f"required_initial_outputs = {self.required_initial_outputs}",
            f"phase_schedules = {self.phase_schedules}",
            f"phase_dependency_edges = {self.phase_dependency_edges}",
        ]
        return "\n".join(lines)


def _format_issue_list(issues: tuple[CompileIssue, ...]) -> tuple[str, ...]:
    return tuple(f"{issue.location}: {issue.message}" for issue in issues)


class CompileError(Exception):
    def __init__(self, report: CompileReport) -> None:
        self.report = report
        messages = "; ".join(f"{issue.location}: {issue.message}" for issue in report.issues)
        super().__init__(f"PRS compile failed: {messages}")


def compile_nodes(
    nodes: Iterable[Node],
    connections: Iterable[Connection] = (),
) -> CompileReport:
    node_tuple = tuple(nodes)
    issues: list[CompileIssue] = []
    connection_map = _connection_map(connections)
    output_paths: list[str] = [
        _node_output_path(node, output)
        for node in node_tuple
        for output in node.__class__._outputs.values()
    ]
    outputs_without_initial = tuple(
        sorted(
            _node_output_path(node, output)
            for node in node_tuple
            for output in node.__class__._outputs.values()
            if output.initial is _MISSING
        )
    )
    output_set = set(output_paths)
    class_output_paths = _class_output_paths(node_tuple)
    inputs: dict[str, str] = {}

    duplicate_outputs = {path for path in output_set if output_paths.count(path) > 1}
    for path in sorted(duplicate_outputs):
        issues.append(
            CompileIssue(
                location=path,
                message="output path is declared more than once",
            )
        )

    node_ids = [node.node_id for node in node_tuple]
    for node_id in sorted({node_id for node_id in set(node_ids) if node_ids.count(node_id) > 1}):
        issues.append(
            CompileIssue(
                location=node_id,
                message="node name is declared more than once",
            )
        )

    for node in node_tuple:
        if node.__class__._input_declaration_error is not None:
            issues.append(
                CompileIssue(
                    location=f"{node.__class__.__name__}.run",
                    message=node.__class__._input_declaration_error,
                )
            )
        for output in node.__class__._outputs.values():
            if output.initial is _MISSING:
                continue
            try:
                output.initial_value(node)
            except Exception as exc:
                issues.append(
                    CompileIssue(
                        location=_node_output_path(node, output),
                        message=f"output initial value failed: {exc}",
                    )
                )

    for node in node_tuple:
        for name, input_port in node.__class__._inputs.items():
            location = _node_input_path(node, input_port)
            source_ref = connection_map.get(location, input_port.source)
            if source_ref is None:
                issues.append(
                    CompileIssue(
                        location=location,
                        message="input source is not connected",
                    )
                )
                continue
            try:
                resolved_source_ref = _resolve_lazy_source(source_ref)
                source = _source_path(source_ref)
            except Exception as exc:
                issues.append(
                    CompileIssue(
                        location=location,
                        message=f"cannot resolve input source: {exc}",
                    )
                )
                continue
            inputs[location] = source
            candidates = class_output_paths.get(source, ())
            is_class_level_output_ref = isinstance(resolved_source_ref, OutputPort)
            if len(candidates) > 1 and (is_class_level_output_ref or source not in output_set):
                issues.append(
                    CompileIssue(
                        location=location,
                        message=(
                            f"ambiguous input source {source!r}; "
                            f"candidates are {tuple(candidates)}; "
                            "use instance connection via port(...).connect(...)"
                        ),
                    )
                )
            elif source not in output_set and source not in SYSTEM_OUTPUTS:
                issues.append(
                    CompileIssue(
                        location=location,
                        message=f"unknown input source {source!r}",
                    )
                )

    return CompileReport(
        nodes=tuple(node.node_id for node in node_tuple),
        inputs=inputs,
        outputs=tuple(sorted(output_set)),
        issues=tuple(issues),
        warnings=(),
        outputs_without_initial=outputs_without_initial,
    )


class PhasedReactiveSystem:
    def __init__(
        self,
        *,
        phases: Iterable[Phase],
        base_dt: BaseTimeStep = "auto",
        connections: Iterable[Connection] = (),
        initial_state: Mapping[Any, Any] | None = None,
        initial_phase: str | None = None,
        initial_tick: int = 0,
        max_phase_steps: int = 64,
        c2star_depth: int | None = None,
        c2star_max_depth: int = 64,
        strict: bool = True,
    ) -> None:
        self.phases = tuple(phases)
        self.nodes = _nodes_from_phases(self.phases)
        _deduplicate_implicit_node_names(self.nodes)
        self.connections = (*_node_connections(self.nodes), *tuple(connections))
        self._connection_map = _connection_map(self.connections)
        self.initial_phase = initial_phase or self._infer_initial_phase()
        self.max_phase_steps = max_phase_steps
        self.c2star_depth = c2star_depth
        self.c2star_max_depth = c2star_max_depth
        self.strict = strict
        self._requested_base_dt = base_dt
        self._continuous_phase_names = tuple(
            phase.name for phase in self.phases if _is_continuous_phase(phase)
        )
        self._base_dt = self._resolve_base_dt(base_dt)
        self._period_ticks = self._resolve_period_ticks()
        self._initial_tick = initial_tick
        self._tick = initial_tick
        self._time_tick = initial_tick
        self._nodes_by_id = {node.node_id: node for node in self.nodes}
        self._nodes_by_type = self._index_nodes_by_type()
        self._outputs_by_path = self._index_outputs()
        self._initial_state_overrides = self._resolve_initial_state(initial_state or {})
        self.compile_report = self._compile()
        if strict and not self.compile_report.ok:
            raise CompileError(self.compile_report)
        self._phases_by_name = {phase.name: phase for phase in self.phases}
        self._phase_schedules = dict(self.compile_report.phase_schedules)
        self._state: dict[str, Any] = {}
        self._history: list[StepRecord] = []
        self.reset()

    @property
    def history(self) -> tuple[StepRecord, ...]:
        return tuple(self._history)

    @property
    def base_dt(self) -> Fraction:
        return self._base_dt

    @property
    def tick(self) -> int:
        return self._tick

    def _resolve_base_dt(self, requested: BaseTimeStep) -> Fraction:
        if requested != "auto":
            return _parse_time_step(requested, field_name="base_dt")
        explicit_dts = self._explicit_schedule_dts()
        if not any(_is_ode_system(node) for node in self.nodes):
            return Fraction(1, 1)
        return _gcd_fraction(explicit_dts)

    def _explicit_schedule_dts(self) -> tuple[Fraction, ...]:
        values: list[Fraction] = []
        for node in self.nodes:
            if _is_ode_system(node):
                values.append(_ode_system_dt(node))
                continue
            node_dt = getattr(node, "_schedule_dt", None)
            if node_dt is not None:
                values.append(node_dt)
        return tuple(values)

    def _resolve_period_ticks(self) -> dict[str, int]:
        periods: dict[str, int] = {}
        for node in self.nodes:
            effective_dt = getattr(node, "_schedule_dt", None) or self._base_dt
            if _is_ode_system(node):
                effective_dt = _ode_system_dt(node)
            ratio = effective_dt / self._base_dt
            if ratio.denominator != 1:
                raise ValueError(
                    f"{node.node_id}.dt={effective_dt} is not an integer multiple "
                    f"of base_dt={self._base_dt}."
                )
            periods[node.node_id] = ratio.numerator
        return periods

    def _is_due(self, node: Node) -> bool:
        return self._tick % self._period_ticks[node.node_id] == 0

    def _clock_state(self) -> dict[str, Any]:
        return {
            f"{SYSTEM_CLOCK}.tick": self._tick,
            f"{SYSTEM_CLOCK}.time": float(self._time_tick * self._base_dt),
        }

    def _commit_clock(self) -> None:
        self._state.update(self._clock_state())

    def reset(
        self,
        initial_state: Mapping[Any, Any] | None = None,
    ) -> None:
        self._state.clear()
        self._history.clear()
        self._tick = self._initial_tick
        self._time_tick = self._initial_tick
        self._reset_ode_runtime_state()
        self._commit_clock()
        for node in self.nodes:
            for output in node.__class__._outputs.values():
                if output.initial is _MISSING:
                    continue
                try:
                    self._state[_node_output_path(node, output)] = output.initial_value(node)
                except Exception:
                    if self.strict:
                        raise
        if initial_state is None:
            self._state.update(self._initial_state_overrides)
        else:
            self._state.update(self._resolve_initial_state(initial_state))
        self._sync_ode_runtime_state_from_system_state()
        self._commit_clock()

    def read(self, output: OutputSource[T]) -> T:
        path = self._resolve_output(output)
        return self._state[path]

    def snapshot(self) -> dict[str, Any]:
        return {path: value for path, value in self._state.items() if path not in SYSTEM_OUTPUTS}

    def _resolve_initial_state(
        self,
        initial_state: Mapping[Any, Any],
    ) -> dict[str, Any]:
        resolved: dict[str, Any] = {}
        for output, value in initial_state.items():
            resolved[self._resolve_output(output)] = value
        return resolved

    def step(self) -> tuple[StepRecord, ...]:
        records: list[StepRecord] = []
        phase_name: str | None = self.initial_phase
        phase_steps = 0
        crossed_continuous_phase = False
        while phase_name is not None:
            phase_steps += 1
            if phase_steps > self.max_phase_steps:
                raise RuntimeError(f"Tick exceeded max_phase_steps={self.max_phase_steps}.")
            phase = self._phases_by_name[phase_name]
            if _is_continuous_phase(phase):
                records.extend(self._run_continuous_phase(phase))
                crossed_continuous_phase = True
            else:
                for node_id in self._phase_schedules[phase.name]:
                    node = self._nodes_by_id[node_id]
                    if not self._is_due(node):
                        continue
                    record = self._run_discrete_node(phase, node)
                    records.append(record)
                    self._history.append(record)
            phase_name = self._choose_next_phase(phase)
        if crossed_continuous_phase:
            self._tick += 1
        else:
            self._tick += 1
            self._time_tick += 1
        self._commit_clock()
        return tuple(records)

    def _run_discrete_node(self, phase: Phase, node: Node) -> StepRecord:
        inputs = self._build_inputs(node)
        result = _run_node(node, inputs, state_snapshot=dict(self._state))
        outputs = self._normalize_outputs(node, result)
        self._commit_node_outputs(node, outputs)
        return StepRecord(
            phase=phase.name,
            node=node.__class__.__name__,
            inputs=dict(vars(inputs)),
            outputs=outputs,
        )

    def _run_continuous_phase(self, phase: Phase) -> tuple[StepRecord, ...]:
        records: list[StepRecord] = []
        time_start = float(self._time_tick * self._base_dt)
        for node in phase.nodes:
            if not self._is_due(node):
                continue
            inputs = self._build_inputs(node)
            result = _run_node(
                node,
                inputs,
                state_snapshot=dict(self._state),
                time_start=time_start,
                time_stop=time_start + float(_ode_system_dt(node)),
            )
            outputs = self._normalize_outputs(node, result)
            self._commit_node_outputs(node, outputs)
            self._commit_ode_state_outputs(node)
            record = StepRecord(
                phase=phase.name,
                node=node.__class__.__name__,
                inputs=dict(vars(inputs)),
                outputs=outputs,
            )
            records.append(record)
            self._history.append(record)
        self._time_tick += 1
        self._commit_clock()
        return tuple(records)

    def _commit_node_outputs(self, node: Node, outputs: dict[str, Any]) -> None:
        for name, value in outputs.items():
            output = node.__class__._outputs[name]
            self._state[_node_output_path(node, output)] = value

    def _commit_ode_state_outputs(self, node: Node) -> None:
        if not _is_ode_system(node):
            return
        for ode_node in _ode_system_nodes(node):
            for name, value in ode_node._ode_state_values.items():
                port = ode_node.__class__._outputs[name]
                self._state[_node_output_path(ode_node, port)] = value

    def _reset_ode_runtime_state(self) -> None:
        time_s = float(self._time_tick * self._base_dt)
        for node in self.nodes:
            if _is_ode_system(node):
                cast(Any, node)._time_s = time_s
                continue
            state_vars = getattr(node.__class__, "_state_vars", {})
            if state_vars:
                cast(Any, node)._ode_state_values = {
                    name: port.initial_value(node) for name, port in state_vars.items()
                }

    def _sync_ode_runtime_state_from_system_state(self) -> None:
        for node in self.nodes:
            state_vars = getattr(node.__class__, "_state_vars", {})
            if not state_vars:
                continue
            values = cast(Any, node)._ode_state_values
            for name, port in state_vars.items():
                path = _node_output_path(node, port)
                if path in self._state:
                    values[name] = self._state[path]

    def run(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

    def compile(self) -> CompileReport:
        return self._compile()

    def _compile(self) -> CompileReport:
        report = compile_nodes(self.nodes, self.connections)
        issues = list(report.issues)
        warnings = list(report.warnings)
        issues.extend(_check_clock_name_is_reserved(self.nodes))
        phase_kind_issues = _check_phase_kinds(self.phases)
        issues.extend(phase_kind_issues)
        if len(self._continuous_phase_names) > 1:
            issues.append(
                CompileIssue(
                    location="phases",
                    message=(
                        "PhasedReactiveSystem supports at most one phase containing ODESystem "
                        "nodes for now"
                    ),
                )
            )
        issues.extend(_check_cross_ode_system_coupling(self.phases, report.inputs))
        warnings.extend(
            _schedule_warnings(
                base_dt=self._base_dt,
                requested_base_dt=self._requested_base_dt,
                continuous=bool(self._continuous_phase_names),
                explicit_dts=tuple(
                    getattr(node, "_schedule_dt")
                    for node in self.nodes
                    if getattr(node, "_schedule_dt", None) is not None and not _is_ode_system(node)
                ),
            )
        )
        completeness_issues = _check_phase_graph_completeness(
            self.phases,
            self.nodes,
            self._connection_map,
        )
        incomplete_source_locations = {
            issue.location for issue in completeness_issues if " reads " in issue.message
        }
        issues = [
            issue
            for issue in issues
            if not (
                issue.location in incomplete_source_locations
                and issue.message.startswith("unknown input source")
            )
        ]
        issues.extend(completeness_issues)
        phase_names = [phase.name for phase in self.phases]
        phase_name_set = set(phase_names)
        initial_phases = [phase.name for phase in self.phases if phase.is_initial]
        phase_schedules: dict[str, tuple[str, ...]] = {}
        phase_dependency_edges: dict[str, tuple[tuple[str, str], ...]] = {}

        for phase_name in sorted(name for name in phase_name_set if phase_names.count(name) > 1):
            issues.append(
                CompileIssue(
                    location=phase_name,
                    message="phase is declared more than once",
                )
            )
        if self.initial_phase not in phase_name_set:
            issues.append(
                CompileIssue(
                    location=self.initial_phase,
                    message="initial phase is not declared",
                )
            )
        if len(initial_phases) != 1:
            issues.append(
                CompileIssue(
                    location="phases",
                    message="exactly one phase must be marked initial",
                )
            )
        elif self.initial_phase != initial_phases[0]:
            issues.append(
                CompileIssue(
                    location=self.initial_phase,
                    message=(
                        f"initial_phase must match the phase marked initial ({initial_phases[0]!r})"
                    ),
                )
            )
        issues.extend(
            _check_c2star(
                self.phases,
                self.nodes,
                depth=self.c2star_depth,
                max_depth=self.c2star_max_depth,
            )
        )
        for phase in self.phases:
            phase_node_ids = self._phase_node_ids(phase)
            if not phase.transitions:
                issues.append(
                    CompileIssue(
                        location=phase.name,
                        message="phase has no transitions",
                    )
                )
            for transition in phase.transitions:
                target = _phase_ref_name(transition.target)
                if target is not None and target not in phase_name_set:
                    issues.append(
                        CompileIssue(
                            location=f"{phase.name}.{transition.name}",
                            message=f"unknown phase target {target!r}",
                        )
                    )
            transition_issues, transition_warnings = _check_transition_structure(phase)
            issues.extend(transition_issues)
            warnings.extend(transition_warnings)
            guard_issues = _check_phase_guard_references(
                phase,
                self._outputs_by_path,
                _class_output_paths(self.nodes),
            )
            issues.extend(
                issue
                for issue in guard_issues
                if not (
                    issue.location in incomplete_source_locations
                    and issue.message.startswith("unknown guard variable")
                )
            )
            if not transition_warnings:
                issues.extend(_check_c3_for_phase(phase, self.nodes))
            dependency_edges = _phase_dependency_edges(phase_node_ids, report.inputs)
            phase_dependency_edges[phase.name] = tuple(dependency_edges)
            schedule = _topological_order(phase_node_ids, dependency_edges)
            if schedule is None:
                edge_text = ", ".join(f"{source}->{target}" for source, target in dependency_edges)
                issues.append(
                    CompileIssue(
                        location=phase.name,
                        message=(f"C1 violation: phase dependency graph is cyclic ({edge_text})"),
                    )
                )
                phase_schedules[phase.name] = phase_node_ids
            else:
                phase_schedules[phase.name] = schedule
        required_initial_outputs = self._required_initial_outputs(
            report.inputs,
            phase_schedules,
        )
        outputs_without_initial = set(report.outputs_without_initial)
        initial_state_outputs = set(self._initial_state_overrides)
        issues.extend(
            _required_initial_issues(
                {
                    path: readers
                    for path, readers in required_initial_outputs.items()
                    if path in outputs_without_initial and path not in initial_state_outputs
                }
            )
        )
        return CompileReport(
            nodes=report.nodes,
            inputs=report.inputs,
            outputs=report.outputs,
            issues=tuple(issues),
            warnings=tuple(warnings),
            phase_schedules=phase_schedules,
            phase_dependency_edges=phase_dependency_edges,
            outputs_without_initial=report.outputs_without_initial,
            required_initial_outputs=required_initial_outputs,
        )

    def _required_initial_outputs(
        self,
        inputs: dict[str, str],
        phase_schedules: dict[str, tuple[str, ...]],
    ) -> dict[str, tuple[str, ...]]:
        issues: dict[str, set[str]] = {}
        phase_by_name = {phase.name: phase for phase in self.phases}
        node_outputs = {
            node.node_id: tuple(
                _node_output_path(node, output) for output in node.__class__._outputs.values()
            )
            for node in self.nodes
        }
        node_inputs = {
            node.node_id: tuple(
                _node_input_path(node, input_port) for input_port in node.__class__._inputs.values()
            )
            for node in self.nodes
        }
        seen: set[tuple[str, frozenset[str]]] = set()
        stack: list[tuple[str, frozenset[str]]] = [(self.initial_phase, frozenset())]

        while stack:
            phase_name, written_before_phase = stack.pop()
            state_key = (phase_name, written_before_phase)
            if state_key in seen:
                continue
            seen.add(state_key)
            phase = phase_by_name.get(phase_name)
            if phase is None:
                continue

            written = set(written_before_phase)
            for node_id in phase_schedules.get(phase.name, self._phase_node_ids(phase)):
                for input_path in node_inputs.get(node_id, ()):
                    source_path = inputs.get(input_path)
                    if (
                        source_path is not None
                        and source_path not in SYSTEM_OUTPUTS
                        and source_path not in written
                    ):
                        issues.setdefault(source_path, set()).add(input_path)
                written.update(node_outputs.get(node_id, ()))

            for transition in phase.transitions:
                target = _phase_ref_name(transition.target)
                if target is not None:
                    stack.append((target, frozenset(written)))

        return {
            source_path: tuple(sorted(readers)) for source_path, readers in sorted(issues.items())
        }

    def _infer_initial_phase(self) -> str:
        initial_phases = [phase.name for phase in self.phases if phase.is_initial]
        if len(initial_phases) == 1:
            return initial_phases[0]
        return "__invalid_initial_phase__"

    def _index_outputs(self) -> dict[str, str]:
        outputs: dict[str, str] = {}
        class_paths: dict[str, list[str]] = {}
        for node in self.nodes:
            for output in node.__class__._outputs.values():
                path = _node_output_path(node, output)
                outputs[path] = path
                class_paths.setdefault(output.path, []).append(path)
        for class_path, paths in class_paths.items():
            if len(paths) == 1:
                outputs[class_path] = paths[0]
        return outputs

    def _index_nodes_by_type(self) -> dict[type[Node], Node]:
        nodes_by_type: dict[type[Node], Node] = {}
        for node in self.nodes:
            if sum(1 for candidate in self.nodes if candidate.__class__ is node.__class__) == 1:
                nodes_by_type[node.__class__] = node
        return nodes_by_type

    def _resolve_node_ref(self, node_ref: NodeRef) -> Node:
        if isinstance(node_ref, Node):
            return node_ref
        try:
            return self._nodes_by_type[node_ref]
        except KeyError as exc:
            raise KeyError(
                f"Phase node reference {node_ref.__name__} is ambiguous or unknown; "
                "use a node instance instead."
            ) from exc

    def _phase_node_ids(self, phase: Phase) -> tuple[str, ...]:
        return tuple(node.node_id for node in phase.nodes)

    def _resolve_output(self, output: OutputSource[T]) -> str:
        output = _resolve_lazy_source(output)
        if isinstance(output, BoundOutputPort):
            return output.path
        if isinstance(output, OutputPort):
            output = output.path
        if isinstance(output, SystemSource):
            output = output.path
        output = _normalize_output_path(output)
        if "." not in output:
            raise ValueError(f"Output reference must be 'Node.output', got {output!r}.")
        if output in SYSTEM_OUTPUTS:
            return output
        try:
            return self._outputs_by_path[output]
        except KeyError as exc:
            raise KeyError(f"Unknown output reference: {output}") from exc

    def _choose_next_phase(self, phase: Phase) -> str | None:
        snapshot = dict(self._state)
        enabled = [
            transition
            for transition in _effective_transitions(phase)
            if _evaluate_guard(transition.predicate, snapshot)
        ]
        if len(enabled) != 1:
            names = [transition.name for transition in enabled]
            raise RuntimeError(
                f"Phase {phase.name!r} must enable exactly one transition; "
                f"enabled={names or ['none']}."
            )
        return _phase_ref_name(enabled[0].target)

    def _build_inputs(self, node: Node) -> NodeInputs:
        values: dict[str, Any] = {}
        for name, input_port in node.__class__._inputs.items():
            input_path = _node_input_path(node, input_port)
            source_ref = self._connection_map.get(input_path, input_port.source)
            if source_ref is None:
                raise RuntimeError(f"Input {input_path} is not connected.")
            output = self._resolve_output(cast(OutputSource[Any], source_ref))
            value = self._state.get(output, input_port.default)
            values[name] = value
        return node.__class__.Inputs(**values)

    def _normalize_outputs(
        self,
        node: Node,
        result: NodeOutputs | dict[str, Any],
    ) -> dict[str, Any]:
        outputs = dict(result) if isinstance(result, dict) else dict(vars(result))
        declared = node.__class__._outputs
        unknown = set(outputs) - set(declared)
        if unknown:
            raise ValueError(
                f"{node.__class__.__name__}.run returned undeclared outputs: {sorted(unknown)}"
            )
        missing = set(declared) - set(outputs)
        if missing:
            raise ValueError(
                f"{node.__class__.__name__}.run did not return outputs: {sorted(missing)}"
            )
        return outputs


def _run_node(
    node: Node,
    inputs: NodeInputs,
    *,
    state_snapshot: StateSnapshot | None = None,
    time_start: float | None = None,
    time_stop: float | None = None,
) -> Any:
    kwargs: dict[str, Any] = {}
    if state_snapshot is not None and _accepts_keyword(node.run, "state_snapshot"):
        kwargs["state_snapshot"] = state_snapshot
    if time_start is not None and _accepts_keyword(node.run, "time_start"):
        kwargs["time_start"] = time_start
    if time_stop is not None and _accepts_keyword(node.run, "time_stop"):
        kwargs["time_stop"] = time_stop
    if node.__class__._run_input_mode == "parameters":
        return node.run(**vars(inputs), **kwargs)
    if node.__class__._inputs or _run_requires_inputs(node):
        return node.run(inputs, **kwargs)
    return node.run(**kwargs)


def _required_initial_issues(
    required_initial_outputs: dict[str, tuple[str, ...]],
) -> list[CompileIssue]:
    return [
        CompileIssue(
            location=source_path,
            message=(f"output initial value is required before first read by {readers}"),
        )
        for source_path, readers in required_initial_outputs.items()
    ]


def _check_phase_graph_completeness(
    phases: tuple[Phase, ...],
    phase_nodes: tuple[Node, ...],
    connection_map: dict[str, OutputSource[Any]],
) -> list[CompileIssue]:
    covered_node_ids = {node.node_id for node in phase_nodes}
    issues: list[CompileIssue] = []

    for node in phase_nodes:
        for input_port in node.__class__._inputs.values():
            location = _node_input_path(node, input_port)
            source_ref = connection_map.get(location, input_port.source)
            if source_ref is None:
                continue
            _append_missing_bound_source_issue(
                issues,
                location,
                source_ref,
                covered_node_ids,
            )

    for phase in phases:
        for transition in phase.transitions:
            for source_ref in _guard_sources(transition.predicate):
                _append_missing_bound_source_issue(
                    issues,
                    f"{phase.name}.{transition.name}",
                    source_ref,
                    covered_node_ids,
                )
    return issues


def _append_missing_bound_source_issue(
    issues: list[CompileIssue],
    location: str,
    source_ref: OutputSource[Any],
    covered_node_ids: set[str],
) -> None:
    try:
        resolved_source_ref = _resolve_lazy_source(source_ref)
    except Exception:
        return
    if not isinstance(resolved_source_ref, BoundOutputPort):
        return
    source_node = resolved_source_ref.node
    if source_node.node_id in covered_node_ids:
        return
    issues.append(
        CompileIssue(
            location=location,
            message=(
                "incomplete phase graph: "
                f"{location} reads {resolved_source_ref.path}, but node "
                f"{source_node.node_id} is not assigned to any phase"
            ),
        )
    )


def _guard_sources(guard: Guard) -> tuple[OutputSource[Any], ...]:
    if isinstance(guard, VarExpr):
        output_source = cast(OutputSource[Any] | None, guard.__dict__.get("output_source"))
        if output_source is not None:
            return (output_source,)
    if isinstance(guard, UnaryExpr):
        return _guard_sources(guard.operand)
    if isinstance(guard, BinaryExpr):
        return (*_guard_sources(guard.left), *_guard_sources(guard.right))
    return ()


def _check_transition_structure(phase: Phase) -> tuple[list[CompileIssue], list[CompileIssue]]:
    issues: list[CompileIssue] = []
    warnings: list[CompileIssue] = []
    transitions = phase.transitions
    if not transitions:
        return issues, warnings

    goto_transitions = [transition for transition in transitions if transition.kind == "goto"]
    if goto_transitions:
        if len(transitions) != 1:
            issues.append(
                CompileIssue(
                    location=phase.name,
                    message="Goto transitions cannot be mixed with If/ElseIf/Else chains",
                )
            )
        return issues, warnings

    has_open_chain = False
    chain_closed = False
    last_else_name: str | None = None
    for transition in transitions:
        if last_else_name is not None and transition.kind == "if":
            warnings.append(
                CompileIssue(
                    location=f"{phase.name}.{transition.name}",
                    message=(
                        f"transition follows Else {last_else_name!r}; "
                        "move it before Else or split the control flow into another phase"
                    ),
                )
            )
        if transition.kind == "if":
            has_open_chain = True
            chain_closed = False
            last_else_name = None
        elif transition.kind == "elseif":
            if not has_open_chain:
                issues.append(
                    CompileIssue(
                        location=f"{phase.name}.{transition.name}",
                        message="ElseIf must follow If or ElseIf",
                    )
                )
            elif chain_closed:
                issues.append(
                    CompileIssue(
                        location=f"{phase.name}.{transition.name}",
                        message="ElseIf must follow If or ElseIf",
                    )
                )
        elif transition.kind == "else":
            if not has_open_chain:
                issues.append(
                    CompileIssue(
                        location=f"{phase.name}.{transition.name}",
                        message="Else must follow If or ElseIf",
                    )
                )
            elif chain_closed:
                issues.append(
                    CompileIssue(
                        location=f"{phase.name}.{transition.name}",
                        message="Else must follow If or ElseIf",
                    )
                )
            chain_closed = True
            last_else_name = transition.name
    return issues, warnings


def _effective_transitions(phase: Phase) -> tuple[EffectiveTransition, ...]:
    return _effective_transition_list(phase.transitions)


def _effective_transition_list(
    transitions: tuple[Transition, ...],
) -> tuple[EffectiveTransition, ...]:
    if len(transitions) == 1:
        transition = transitions[0]
        if transition.kind == "goto":
            return (
                EffectiveTransition(
                    predicate=ConstExpr(True),
                    target=transition.target,
                    name=transition.name,
                ),
            )

    effective: list[EffectiveTransition] = []
    previous_in_chain: list[Guard] = []
    chain_closed = False
    has_open_chain = False
    for transition in transitions:
        if transition.kind == "goto":
            return (
                EffectiveTransition(
                    predicate=ConstExpr(True),
                    target=transition.target,
                    name=transition.name,
                ),
            )
        if transition.kind == "if":
            previous_in_chain = []
            chain_closed = False
            has_open_chain = True
            predicate = transition.predicate
            previous_in_chain.append(transition.predicate)
        elif transition.kind == "elseif" and has_open_chain and not chain_closed:
            predicate = _guard_after_previous_failed(
                transition.predicate,
                tuple(previous_in_chain),
            )
            previous_in_chain.append(transition.predicate)
        elif transition.kind == "else" and has_open_chain and not chain_closed:
            predicate = _all_previous_guards_failed(tuple(previous_in_chain))
            chain_closed = True
        else:
            predicate = transition.predicate

        effective.append(
            EffectiveTransition(
                predicate=predicate,
                target=transition.target,
                name=transition.name,
            )
        )
    return tuple(effective)


def _guard_after_previous_failed(predicate: Guard, previous: tuple[Guard, ...]) -> Guard:
    if isinstance(predicate, Expr) and all(isinstance(guard, Expr) for guard in previous):
        return _all_previous_exprs_failed(cast(tuple[Expr, ...], previous)) & predicate

    def guard(state: StateSnapshot) -> bool:
        return all(
            not _evaluate_guard(previous_guard, state) for previous_guard in previous
        ) and _evaluate_guard(predicate, state)

    return guard


def _all_previous_guards_failed(previous: tuple[Guard, ...]) -> Guard:
    if all(isinstance(guard, Expr) for guard in previous):
        return _all_previous_exprs_failed(cast(tuple[Expr, ...], previous))

    def guard(state: StateSnapshot) -> bool:
        return all(not _evaluate_guard(previous_guard, state) for previous_guard in previous)

    return guard


def _all_previous_exprs_failed(previous: tuple[Expr, ...]) -> Expr:
    if not previous:
        return ConstExpr(True)
    expr = previous[0]
    for guard in previous[1:]:
        expr = expr | guard
    return ~expr


def _check_phase_guard_references(
    phase: Phase,
    outputs_by_path: dict[str, str],
    class_output_paths: dict[str, tuple[str, ...]],
) -> list[CompileIssue]:
    issues: list[CompileIssue] = []
    for transition in phase.transitions:
        if not isinstance(transition.predicate, Expr):
            continue
        for variable in sorted(transition.predicate.variables):
            try:
                _resolve_guard_variable_path(
                    variable,
                    outputs_by_path,
                    class_output_paths,
                )
            except ValueError as exc:
                issues.append(
                    CompileIssue(
                        location=f"{phase.name}.{transition.name}",
                        message=str(exc),
                    )
                )
                continue
    return issues


def _resolve_guard_variable_path(
    variable: str,
    outputs_by_path: dict[str, str],
    class_output_paths: dict[str, tuple[str, ...]],
) -> str:
    variable = _normalize_output_path(variable)
    if variable in SYSTEM_OUTPUTS:
        return variable
    try:
        return outputs_by_path[variable]
    except KeyError:
        candidates = class_output_paths.get(variable, ())
        if len(candidates) > 1:
            raise ValueError(
                f"ambiguous guard variable {variable!r}; "
                f"candidates are {tuple(candidates)}; use instance output reference"
            )
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError(f"unknown guard variable {variable!r}")


def _run_requires_inputs(node: Node) -> bool:
    try:
        signature = inspect.signature(node.run)
    except (TypeError, ValueError):
        return True
    return any(
        parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        for parameter in signature.parameters.values()
    )


def _connection_map(
    connections: Iterable[Connection],
) -> dict[str, OutputSource[Any]]:
    return {connection.input.path: connection.source for connection in connections}


def _node_connections(nodes: Iterable[Node]) -> tuple[Connection, ...]:
    return tuple(connection for node in nodes for connection in node._connections.values())


def _nodes_from_phases(phases: tuple[Phase, ...]) -> tuple[Node, ...]:
    nodes: list[Node] = []
    seen: set[int] = set()
    for phase in phases:
        for node in phase.nodes:
            for expanded_node in _expand_phase_node(node):
                identity = id(expanded_node)
                if identity in seen:
                    continue
                nodes.append(expanded_node)
                seen.add(identity)
    return tuple(nodes)


def _expand_phase_node(node: Node) -> tuple[Node, ...]:
    if _is_ode_system(node):
        return (node, *_ode_system_nodes(node))
    return (node,)


def _is_ode_system(node: Node) -> bool:
    return bool(getattr(node, "_is_ode_system", False))


def _ode_system_dt(node: Node) -> Fraction:
    return cast(Any, node).dt


def _ode_system_nodes(node: Node) -> tuple[Any, ...]:
    return tuple(cast(Any, node).nodes)


def _is_continuous_phase(phase: Phase) -> bool:
    return bool(phase.nodes) and all(_is_ode_system(node) for node in phase.nodes)


def _is_discrete_phase(phase: Phase) -> bool:
    return all(not _is_ode_system(node) for node in phase.nodes)


def _check_phase_kinds(phases: tuple[Phase, ...]) -> list[CompileIssue]:
    issues: list[CompileIssue] = []
    for phase in phases:
        if _is_continuous_phase(phase) or _is_discrete_phase(phase):
            continue
        issues.append(
            CompileIssue(
                location=phase.name,
                message="phase cannot mix ODESystem nodes with ordinary nodes",
            )
        )
    return issues


def _check_cross_ode_system_coupling(
    phases: tuple[Phase, ...],
    inputs: dict[str, str],
) -> list[CompileIssue]:
    issues: list[CompileIssue] = []
    for phase in phases:
        if not _is_continuous_phase(phase) or len(phase.nodes) < 2:
            continue
        ode_system_by_node_id: dict[str, str] = {}
        for ode_system in phase.nodes:
            for ode_node in _ode_system_nodes(ode_system):
                ode_system_by_node_id[ode_node.node_id] = ode_system.node_id
        for ode_system in phase.nodes:
            ode_system_id = ode_system.node_id
            for ode_node in _ode_system_nodes(ode_system):
                for input_port in ode_node.__class__._inputs.values():
                    input_path = _node_input_path(ode_node, input_port)
                    source_path = inputs.get(input_path)
                    if source_path is None:
                        continue
                    source_node_id = source_path.split(".", maxsplit=1)[0]
                    source_system_id = ode_system_by_node_id.get(source_node_id)
                    if source_system_id is None or source_system_id == ode_system_id:
                        continue
                    issues.append(
                        CompileIssue(
                            location=input_path,
                            message=(
                                "continuous phase contains coupled ODESystem nodes: "
                                f"{input_path} reads {source_path} across ODESystem "
                                f"boundary {source_system_id!r} -> {ode_system_id!r}. "
                                "Put continuously coupled ODENodes into the same ODESystem."
                            ),
                        )
                    )
    return issues


def _check_clock_name_is_reserved(nodes: tuple[Node, ...]) -> list[CompileIssue]:
    return [
        CompileIssue(
            location=node.node_id,
            message="Clock is a reserved system source name",
        )
        for node in nodes
        if node.node_id == SYSTEM_CLOCK
    ]


def _schedule_warnings(
    *,
    base_dt: Fraction,
    requested_base_dt: BaseTimeStep,
    continuous: bool,
    explicit_dts: tuple[Fraction, ...],
) -> list[CompileIssue]:
    if continuous or not explicit_dts:
        return []
    common = _gcd_fraction(explicit_dts)
    if common <= base_dt:
        return []
    if requested_base_dt == "auto":
        message = (
            f'discrete-only system uses base_dt=1 for base_dt="auto", but all explicit '
            f"node dt values are multiples of {common}; this creates idle ticks. "
            f"Set base_dt={common!s} to compress the schedule."
        )
    else:
        message = (
            f"explicit base_dt={base_dt} creates idle ticks because all explicit node dt "
            f"values are multiples of {common}. Set base_dt={common!s} if this is not "
            "intentional."
        )
    return [CompileIssue(location="base_dt", message=message)]


def _phase_ref_name(phase_ref: PhaseRef) -> str | None:
    if phase_ref is None or phase_ref is terminate:
        return None
    if isinstance(phase_ref, Phase):
        return phase_ref.name
    if isinstance(phase_ref, TerminateTarget):
        return None
    return phase_ref


def _class_output_paths(nodes: tuple[Node, ...]) -> dict[str, tuple[str, ...]]:
    paths: dict[str, list[str]] = {}
    for node in nodes:
        for output in node.__class__._outputs.values():
            paths.setdefault(output.path, []).append(_node_output_path(node, output))
    return {path: tuple(candidates) for path, candidates in paths.items()}


def _deduplicate_implicit_node_names(nodes: tuple[Node, ...]) -> None:
    used: set[str] = set()
    implicit_counts: dict[str, int] = {}
    for node in nodes:
        base = node.node_id
        if node._name_is_explicit:
            used.add(base)
            continue
        implicit_counts[base] = implicit_counts.get(base, 0) + 1
        suffix = implicit_counts[base]
        candidate = base if suffix == 1 else f"{base}_{suffix}"
        while candidate in used:
            suffix += 1
            implicit_counts[base] = suffix
            candidate = f"{base}_{suffix}"
        node.node_id = candidate
        node.name = candidate
        used.add(candidate)


def _node_output_path(node: Node, output: OutputPort[Any]) -> str:
    if output.name is None:
        return "<unbound>"
    return f"{node.node_id}.{output.name}"


def _node_input_path(node: Node, input_port: InputPort[Any]) -> str:
    if input_port.name is None:
        return "<unbound>"
    return f"{node.node_id}.{input_port.name}"


def _source_path(source: OutputSource[Any]) -> str:
    source = _resolve_lazy_source(source)
    if isinstance(source, BoundOutputPort):
        return source.path
    if isinstance(source, OutputPort):
        return source.path
    if isinstance(source, SystemSource):
        return source.path
    if not isinstance(source, str):
        raise TypeError(
            "Output source must be an OutputPort, SystemSource, a string reference, "
            "or a zero-argument callable returning one."
        )
    return _normalize_output_path(source)


def _resolve_lazy_source(source: OutputSource[T]) -> ResolvedOutputSource[T]:
    if isinstance(source, (BoundOutputPort, OutputPort, SystemSource, str)):
        return source
    return source()


def _normalize_output_path(path: str) -> str:
    parts = path.split(".")
    if len(parts) == 3:
        return f"{parts[0]}.{parts[2]}"
    return path


def _phase_dependency_edges(
    phase_node_ids: tuple[str, ...],
    inputs: dict[str, str],
) -> list[tuple[str, str]]:
    active = set(phase_node_ids)
    edges: set[tuple[str, str]] = set()
    for target_node in active:
        prefix = f"{target_node}."
        for input_path, source_path in inputs.items():
            if not input_path.startswith(prefix):
                continue
            source_node = source_path.split(".", maxsplit=1)[0]
            if source_node in active and source_node != target_node:
                edges.add((source_node, target_node))
    return sorted(edges)


def _node_ref_outputs(node_ref: NodeRef) -> Iterable[tuple[str, OutputPort[Any]]]:
    for output in node_ref.__class__._outputs.values():
        yield _node_output_path(node_ref, output), output


def _has_cycle(nodes: list[str], edges: list[tuple[str, str]]) -> bool:
    return _topological_order(tuple(nodes), edges) is None


def _topological_order(
    nodes: tuple[str, ...],
    edges: list[tuple[str, str]],
) -> tuple[str, ...] | None:
    node_set = set(nodes)
    order_index = {node: index for index, node in enumerate(nodes)}
    adjacency: dict[str, list[str]] = {node: [] for node in nodes}
    indegree: dict[str, int] = {node: 0 for node in nodes}
    for source, target in edges:
        if source not in node_set or target not in node_set:
            continue
        adjacency.setdefault(source, []).append(target)
        indegree[target] = indegree.get(target, 0) + 1
    for targets in adjacency.values():
        targets.sort(key=order_index.__getitem__)

    queue = [node for node in nodes if indegree.get(node, 0) == 0]
    schedule: list[str] = []
    while queue:
        node = queue.pop(0)
        schedule.append(node)
        for target in adjacency.get(node, []):
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
                queue.sort(key=order_index.__getitem__)
    if len(schedule) != len(nodes):
        return None
    return tuple(schedule)


def _check_c2star(
    phases: tuple[Phase, ...],
    nodes: tuple[Node, ...],
    *,
    depth: int | None,
    max_depth: int,
) -> list[CompileIssue]:
    phase_names = [phase.name for phase in phases]
    edges = _phase_transition_edges(phases)
    if not _has_cycle(phase_names, edges):
        return []

    phase_map = {phase.name: phase for phase in phases}
    output_types = _output_types(nodes)
    domains = _finite_domains_by_path(nodes)
    issues: list[CompileIssue] = []
    for cycle in _simple_cycles(phase_names, edges):
        result = _check_c2star_for_cycle(
            cycle,
            phase_map,
            output_types,
            domains,
            depth=depth,
            max_depth=max_depth,
        )
        if result is not None:
            issues.append(result)
    return issues


def _phase_transition_edges(phases: tuple[Phase, ...]) -> list[tuple[str, str]]:
    return sorted(
        (phase.name, target)
        for phase in phases
        for transition in _effective_transitions(phase)
        for target in (_phase_ref_name(transition.target),)
        if target is not None
    )


def _simple_cycles(
    nodes: list[str],
    edges: list[tuple[str, str]],
) -> tuple[tuple[str, ...], ...]:
    adjacency: dict[str, list[str]] = {node: [] for node in nodes}
    for source, target in edges:
        adjacency.setdefault(source, []).append(target)

    cycles: set[tuple[str, ...]] = set()
    for start in nodes:
        stack: list[tuple[str, list[str]]] = [(start, [start])]
        while stack:
            current, path = stack.pop()
            for target in adjacency.get(current, []):
                if target == start:
                    cycles.add(_canonical_cycle(tuple(path)))
                elif target not in path:
                    stack.append((target, [*path, target]))
    return tuple(sorted(cycles))


def _canonical_cycle(cycle: tuple[str, ...]) -> tuple[str, ...]:
    rotations = [cycle[index:] + cycle[:index] for index in range(len(cycle))]
    return min(rotations)


def _check_c2star_for_cycle(
    cycle: tuple[str, ...],
    phases: dict[str, Phase],
    output_types: dict[str, type[Any]],
    domains: dict[str, tuple[Any, ...]],
    *,
    depth: int | None,
    max_depth: int,
) -> CompileIssue | None:
    guards: list[Expr] = []
    for index, phase_name in enumerate(cycle):
        next_phase = cycle[(index + 1) % len(cycle)]
        guard = _cycle_guard(phases[phase_name], next_phase)
        if guard is None:
            return CompileIssue(
                location=" -> ".join((*cycle, cycle[0])),
                message=(
                    "C2 violation: phase graph contains a cycle, "
                    "and C2* cannot check non-symbolic guards"
                ),
            )
        guards.append(guard)

    cycle_writes = frozenset(
        output_path
        for phase_name in cycle
        for node_ref in phases[phase_name].nodes
        for output_path, _ in _node_ref_outputs(node_ref)
    )
    guard_vars = frozenset().union(*(guard.variables for guard in guards))
    relevant_vars = guard_vars & cycle_writes
    traversal_count = depth
    if traversal_count is None:
        bound = _c2star_collapse_bound(relevant_vars, domains)
        if bound is None:
            return CompileIssue(
                location=" -> ".join((*cycle, cycle[0])),
                message=(
                    "C2 violation: phase graph contains a cycle, "
                    "and C2* exact bound needs finite domains for "
                    f"{sorted(relevant_vars)}"
                ),
            )
        traversal_count = bound
    if traversal_count > max_depth:
        return CompileIssue(
            location=" -> ".join((*cycle, cycle[0])),
            message=(
                "C2 violation: phase graph contains a cycle, "
                f"and C2*({traversal_count}) exceeds max_depth={max_depth} "
                f"for R_C={sorted(relevant_vars)}"
            ),
        )

    solver = z3.Solver()
    state = {
        path: _z3_variable_for_type(output_types[path], f"initial::{path}") for path in guard_vars
    }
    all_bindings = dict(state)
    initial_ctx = Z3Context(output_types, domains=domains, bindings=state)
    solver.add(*initial_ctx.domain_constraints(guard_vars))
    for traversal in range(1, traversal_count + 1):
        for phase_index, phase_name in enumerate(cycle, start=1):
            phase_writes = {
                output_path
                for node_ref in phases[phase_name].nodes
                for output_path, _ in _node_ref_outputs(node_ref)
            }
            for path in sorted(phase_writes & guard_vars):
                variable = _z3_variable_for_type(
                    output_types[path],
                    f"write::{traversal}::{phase_index}::{path}",
                )
                state[path] = variable
                all_bindings[f"{path}@{traversal}.{phase_index}"] = variable
            ctx = Z3Context(output_types, domains=domains, bindings=dict(state))
            solver.add(*ctx.domain_constraints(phase_writes & guard_vars))
            solver.add(guards[phase_index - 1].to_z3(ctx))

    if solver.check() != z3.sat:
        return None
    model_ctx = Z3Context(output_types, domains=domains, bindings=all_bindings)
    return CompileIssue(
        location=" -> ".join((*cycle, cycle[0])),
        message=(
            f"C2*({traversal_count}) violation: cycle is feasible for "
            f"{traversal_count} traversal(s), R_C={sorted(relevant_vars)}, "
            f"witness={_model_snapshot(solver.model(), model_ctx, limit=16)}"
        ),
    )


def _cycle_guard(phase: Phase, target: str) -> Expr | None:
    guards = [
        transition.predicate
        for transition in _effective_transitions(phase)
        if _phase_ref_name(transition.target) == target
    ]
    if not guards or not all(isinstance(guard, Expr) for guard in guards):
        return None
    expr = cast(Expr, guards[0])
    for guard in guards[1:]:
        expr = expr | cast(Expr, guard)
    return expr


def _c2star_collapse_bound(
    relevant_vars: frozenset[str],
    domains: dict[str, tuple[Any, ...]],
) -> int | None:
    bound = 1
    for path in relevant_vars:
        domain = domains.get(path)
        if domain is None:
            return None
        bound *= len(domain)
    return bound


def _z3_variable_for_type(output_type: type[Any], name: str) -> Any:
    if output_type is bool:
        return z3.Bool(name)
    if output_type is int:
        return z3.Int(name)
    if output_type is float:
        return z3.Real(name)
    if _is_enum_type(output_type):
        return z3.Int(name)
    raise TypeError(f"Output type {output_type!r} is not supported by z3.")


def _z3_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return z3.IntVal(_enum_member_index(value))
    if isinstance(value, bool):
        return z3.BoolVal(value)
    if isinstance(value, int):
        return z3.IntVal(value)
    if isinstance(value, float):
        return z3.RealVal(value)
    raise TypeError(f"Value {value!r} is not supported by z3.")


def _is_enum_type(value: Any) -> TypeGuard[type[Enum]]:
    return isinstance(value, type) and issubclass(value, Enum)


def _enum_member_index(value: Enum) -> int:
    return list(type(value)).index(value)


def _evaluate_guard(predicate: Guard, state: StateSnapshot) -> bool:
    if isinstance(predicate, Expr):
        return predicate.evaluate(state)
    return predicate(state)


def _check_c3_for_phase(
    phase: Phase,
    nodes: tuple[Node, ...],
    *,
    max_states: int = 4096,
) -> list[CompileIssue]:
    if not phase.transitions:
        return []
    effective_transitions = _effective_transitions(phase)
    if all(isinstance(transition.predicate, Expr) for transition in effective_transitions):
        return _check_c3_for_phase_with_z3(phase, nodes, effective_transitions)
    issues: list[CompileIssue] = []
    samples = _state_samples(nodes, max_states=max_states)
    if samples is None:
        return [
            CompileIssue(
                location=phase.name,
                message=f"C3 check skipped: finite state sample exceeds {max_states}",
            )
        ]

    for state in samples:
        enabled: list[str] = []
        for transition in effective_transitions:
            try:
                if _evaluate_guard(transition.predicate, state):
                    enabled.append(transition.name)
            except KeyError:
                return []
            except Exception as exc:
                issues.append(
                    CompileIssue(
                        location=f"{phase.name}.{transition.name}",
                        message=f"C3 predicate failed during compile check: {exc}",
                    )
                )
                continue
        if len(enabled) != 1:
            issues.append(
                CompileIssue(
                    location=phase.name,
                    message=(
                        "C3 violation: expected exactly one enabled transition, "
                        f"got {enabled or ['none']} at state {state}"
                    ),
                )
            )
            break
    return issues


def _check_c3_for_phase_with_z3(
    phase: Phase,
    nodes: tuple[Node, ...],
    transitions: tuple[EffectiveTransition, ...],
) -> list[CompileIssue]:
    output_types = _output_types(nodes)
    output_types[f"{SYSTEM_CLOCK}.tick"] = int
    output_types[f"{SYSTEM_CLOCK}.time"] = float
    domains = _finite_domains_by_path(nodes)
    ctx = Z3Context(output_types, domains=domains)
    try:
        predicates = [cast(Expr, transition.predicate).to_z3(ctx) for transition in transitions]
    except Exception as exc:
        return [
            CompileIssue(
                location=phase.name,
                message=f"C3 z3 check failed: {exc}",
            )
        ]

    issues: list[CompileIssue] = []
    guard_vars = frozenset().union(
        *[cast(Expr, transition.predicate).variables for transition in transitions]
    )
    for left_index, left in enumerate(predicates):
        for right_index, right in enumerate(predicates[left_index + 1 :], left_index + 1):
            solver = z3.Solver()
            solver.add(*ctx.domain_constraints(guard_vars))
            solver.add(left)
            solver.add(right)
            if solver.check() == z3.sat:
                left_transition = transitions[left_index]
                right_transition = transitions[right_index]
                issues.append(
                    CompileIssue(
                        location=phase.name,
                        message=(
                            "C3 violation: transitions "
                            f"{left_transition.name!r} and {right_transition.name!r} "
                            f"overlap at state {_model_snapshot(solver.model(), ctx)}"
                        ),
                    )
                )
                return issues

    solver = z3.Solver()
    solver.add(*ctx.domain_constraints(guard_vars))
    solver.add(z3.Not(z3.Or(*predicates)))
    if solver.check() == z3.sat:
        issues.append(
            CompileIssue(
                location=phase.name,
                message=(
                    "C3 violation: no transition is enabled at state "
                    f"{_model_snapshot(solver.model(), ctx)}"
                ),
            )
        )
    return issues


def _model_snapshot(
    model: z3.ModelRef,
    ctx: Z3Context,
    *,
    limit: int | None = None,
) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    variables = ctx.variables | ctx.bindings
    for index, (path, variable) in enumerate(sorted(variables.items())):
        if limit is not None and index >= limit:
            snapshot["..."] = f"{len(variables) - limit} more"
            break
        value = model.eval(variable, model_completion=True)
        snapshot[path] = str(value)
    return snapshot


def _output_types(nodes: tuple[Node, ...]) -> dict[str, type[Any]]:
    types: dict[str, type[Any]] = {}
    for node in nodes:
        node_cls = node.__class__
        module = sys.modules[node_cls.__module__]
        try:
            hints = get_type_hints(
                node_cls._output_namespace_cls,
                globalns=vars(module),
                localns={node_cls.__name__: node_cls},
                include_extras=True,
            )
        except Exception:
            hints = {}
        for name, output in node_cls._outputs.items():
            annotation = hints.get(name)
            types[_node_output_path(node, output)] = _base_type(annotation)
    return types


def _finite_domains_by_path(nodes: tuple[Node, ...]) -> dict[str, tuple[Any, ...]]:
    domains: dict[str, tuple[Any, ...]] = {}
    output_types = _output_types(nodes)
    for node in nodes:
        for output in node.__class__._outputs.values():
            path = _node_output_path(node, output)
            domain = output.finite_domain(node)
            if domain is None:
                output_type = output_types.get(path)
                if _is_enum_type(output_type):
                    domain = tuple(output_type)
            if domain is not None:
                domains[path] = domain
    return domains


def _base_type(annotation: Any) -> type[Any]:
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    if annotation in (bool, int, float):
        return annotation
    if _is_enum_type(annotation):
        return annotation
    return object


def _state_samples(
    nodes: tuple[Node, ...],
    *,
    max_states: int,
) -> list[StateSnapshot] | None:
    entries: list[tuple[str, tuple[Any, ...]]] = []
    state_count = 1
    domains_by_path = _finite_domains_by_path(nodes)
    for node in nodes:
        for output in node.__class__._outputs.values():
            path = _node_output_path(node, output)
            domain = domains_by_path.get(path)
            if domain is None:
                continue
            state_count *= len(domain)
            if state_count > max_states:
                return None
            entries.append((path, domain))

    keys = [key for key, _ in entries]
    domains = [domain for _, domain in entries]
    return [dict(zip(keys, values, strict=True)) for values in product(*domains)]
