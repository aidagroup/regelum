from __future__ import annotations

import inspect
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from functools import reduce
from math import gcd
from typing import (
    Annotated,
    Any,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeGuard,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

import z3

if sys.version_info >= (3, 11):
    from typing import dataclass_transform
else:
    from typing_extensions import dataclass_transform

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
    var_source: VarSource[Any] | None = None

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
            raise TypeError(f"State variable {path!r} has no z3-supported type.")
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


def V(var: Any) -> Expr:
    return VarExpr(_source_path(var), var_source=var)


Guard: TypeAlias = Predicate | Expr


class VarPort(Generic[T]):
    def __init__(
        self,
        initial: InitialValue = _MISSING,
        domain: Iterable[T] | None = None,
    ) -> None:
        self.initial = initial
        self.domain = tuple(domain) if domain is not None else None
        self.name: str | None = None
        self.node_cls: type[Node] | None = None

    def __set_name__(self, owner: type[NodeState], name: str) -> None:
        self.name = name

    @overload
    def __get__(self, instance: None, owner: type[NodeState]) -> VarPort[T]: ...

    @overload
    def __get__(self, instance: NodeState, owner: type[NodeState]) -> T: ...

    def __get__(
        self,
        instance: NodeState | None,
        owner: type[NodeState],
    ) -> VarPort[T] | T:
        if instance is None:
            return self
        if self.name is None:
            raise AttributeError("Variable is not bound to an attribute name.")
        return instance.__dict__[self.name]

    def initial_value(self, node: Node) -> Any:
        if self.initial is _MISSING:
            raise RuntimeError(f"Variable {self.path} does not define an initial value.")
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


@dataclass(frozen=True)
class BoundVarPort(Generic[T]):
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


ResolvedVarSource: TypeAlias = VarPort[T] | BoundVarPort[T] | SystemSource | str
VarSource: TypeAlias = ResolvedVarSource[T] | Callable[[], ResolvedVarSource[T]]


@dataclass(frozen=True)
class BoundInputPort(Generic[T]):
    node: Node
    port: Any

    @property
    def path(self) -> str:
        if self.port.name is None:
            return "<unbound>"
        return f"{self.node.node_id}.{self.port.name}"

    def connect(self, var: Any) -> Connection:
        connection = connect(self, var)
        self.node._connections[self.path] = connection
        return connection


@dataclass(frozen=True)
class Connection:
    input: BoundInputPort[Any]
    source: Any


def connect(
    input_port: Any,
    var: Any,
) -> Connection:
    if not isinstance(input_port, BoundInputPort):
        raise TypeError(f"connect(...) expects an input port on the left side; got {input_port!r}.")
    if not isinstance(var, (BoundVarPort, VarPort, SystemSource, str)):
        raise TypeError(
            "connect(...) expects a state port, state reference, or system source on "
            "the right side; "
            f"got {var!r}."
        )
    return Connection(input=input_port, source=var)


def port(reference: Any) -> ConnectablePort:
    return cast(ConnectablePort, reference)


class InputPort(Generic[T]):
    def __init__(
        self,
        src: VarSource[T] | None = None,
        default: T | None = None,
    ) -> None:
        self.source = src
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
            raise AttributeError("src is not bound to an attribute name.")
        return instance.__dict__.get(self.name, self.default)

    @property
    def path(self) -> str:
        if self.node_cls is None or self.name is None:
            return "<unbound>"
        return f"{self.node_cls.__name__}.{self.name}"

    def __repr__(self) -> str:
        return self.path


def var(
    *,
    init: InitialValue = _MISSING,
    domain: Iterable[Any] | None = None,
) -> Any:
    return VarPort(initial=init, domain=domain)


def src(
    src: Any = None,
    *,
    default: Any = None,
) -> Any:
    return InputPort(src=src, default=default)


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
    raise TypeError("var init callable must accept zero arguments or one node argument.")


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


@dataclass_transform(field_specifiers=(src,))
class NodeInputs:
    def __init__(self, **values: Any) -> None:
        for name, value in values.items():
            setattr(self, name, value)


@dataclass_transform(field_specifiers=(var,))
class NodeState:
    def __init__(self, **values: Any) -> None:
        for name, value in values.items():
            setattr(self, name, value)


class _BoundPortNamespace:
    def __init__(
        self,
        node: Node,
        nested_cls: type[NodeInputs] | type[NodeState],
        ports: dict[str, InputPort[Any]] | dict[str, VarPort[Any]],
        bound_type: type[BoundInputPort[Any]] | type[BoundVarPort[Any]],
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
    State = NodeState

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        input_namespace_name, input_namespace_cls = _find_port_namespace(
            cls,
            NodeInputs,
            "input",
        )
        output_namespace_name, output_namespace_cls = _find_port_namespace(
            cls,
            NodeState,
            "state",
        )
        if input_namespace_name is not None:
            _install_annotated_ports(cls, input_namespace_cls, InputPort)
        if output_namespace_name is not None:
            _install_annotated_ports(cls, output_namespace_cls, VarPort)
        nested_inputs = _collect_ports(input_namespace_cls, InputPort)
        update_inputs = _collect_update_input_ports(cls)
        update_state_parameters = _collect_update_state_parameter_names(
            cls,
            input_namespace_name,
            cast(type[NodeInputs] | None, input_namespace_cls),
            output_namespace_name,
            cast(type[NodeState] | None, output_namespace_cls),
        )
        cls._input_declaration_error: str | None = None
        cls._input_namespace_name = input_namespace_name
        cls._input_namespace_cls = input_namespace_cls
        cls._output_namespace_name = output_namespace_name
        cls._output_namespace_cls = output_namespace_cls
        cls._state_namespace_name = output_namespace_name
        cls._state_namespace_cls = output_namespace_cls
        cls._update_state_parameter_names = update_state_parameters
        if nested_inputs and update_inputs:
            cls._input_declaration_error = "define inputs either as a NodeInputs namespace or as update(...) parameters, not both"
            cls._run_input_mode = "object"
            cls._inputs = nested_inputs
        elif update_inputs:
            inputs_cls = type("Inputs", (NodeInputs,), dict(update_inputs))
            for name, port in update_inputs.items():
                port.__set_name__(inputs_cls, name)
            cls.Inputs = inputs_cls  # ty: ignore[invalid-assignment]
            cls._input_namespace_name = "Inputs"
            cls._input_namespace_cls = inputs_cls
            cls._run_input_mode = "parameters"
            cls._inputs = update_inputs
        else:
            cls._run_input_mode = "object" if nested_inputs else "none"
            cls._inputs = nested_inputs
        cls._outputs = _collect_ports(output_namespace_cls, VarPort)
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
        class_dt = self.__class__.__dict__.get("dt") if dt is None else None
        schedule_dt = dt if dt is not None else class_dt
        self._schedule_dt = (
            _parse_time_step(schedule_dt, field_name=f"{self.node_id}.dt")
            if schedule_dt is not None
            else None
        )
        if self._schedule_dt is not None and not getattr(
            self.__class__,
            "_allow_schedule_dt",
            True,
        ):
            raise TypeError(
                f"{self.__class__.__name__} cannot define dt; set dt on ODESystem instead."
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
                    cast(type[NodeState], self.__class__._output_namespace_cls),
                    self.__class__._outputs,
                    BoundVarPort,
                ),
            )

    def update(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def _find_port_namespace(
    node_cls: type[Node],
    namespace_base: type[NodeInputs] | type[NodeState],
    namespace_role: str,
) -> tuple[str | None, type[NodeInputs] | type[NodeState] | None]:
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
    nested_cls: type[NodeInputs] | type[NodeState] | None,
    port_type: type[InputPort[Any]] | type[VarPort[Any]],
) -> dict[str, Any]:
    if nested_cls is None:
        return {}
    return {name: value for name, value in vars(nested_cls).items() if isinstance(value, port_type)}


def _collect_update_input_ports(node_cls: type[Node]) -> dict[str, InputPort[Any]]:
    update = node_cls.__dict__.get("update")
    if update is None:
        return {}
    try:
        signature = inspect.signature(update)
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


def _collect_update_state_parameter_names(
    node_cls: type[Node],
    input_namespace_name: str | None,
    input_namespace_cls: type[NodeInputs] | None,
    state_namespace_name: str | None,
    state_cls: type[NodeState] | None,
) -> tuple[str, ...]:
    update = node_cls.__dict__.get("update")
    if update is None or state_cls is None:
        return ()
    try:
        signature = inspect.signature(update)
    except (TypeError, ValueError):
        return ()
    module = sys.modules[node_cls.__module__]
    localns: dict[str, Any] = {node_cls.__name__: node_cls}
    if input_namespace_name is not None and input_namespace_cls is not None:
        localns[input_namespace_name] = input_namespace_cls
    if state_namespace_name is not None:
        localns[state_namespace_name] = state_cls
    localns["State"] = state_cls
    try:
        hints = get_type_hints(
            update,
            globalns=vars(module),
            localns=localns,
            include_extras=True,
        )
    except Exception:
        hints = getattr(update, "__annotations__", {})
    state_parameters: list[str] = []
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue
        if isinstance(parameter.default, InputPort):
            continue
        annotation = hints.get(name)
        if annotation is state_cls or annotation in {
            state_namespace_name,
            "State",
            f"{node_cls.__name__}.{state_namespace_name}",
            f"{node_cls.__name__}.State",
        }:
            state_parameters.append(name)
    return tuple(state_parameters)


def _install_annotated_ports(
    node_cls: type[Node],
    nested_cls: type[NodeInputs] | type[NodeState] | None,
    port_type: type[InputPort[Any]] | type[VarPort[Any]],
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
    port_type: type[InputPort[Any]] | type[VarPort[Any]],
) -> InputPort[Any] | VarPort[Any] | None:
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
    state: dict[str, Any]


NodeRef: TypeAlias = Node
TransitionKind: TypeAlias = Literal["if", "elseif", "else", "goto"]


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


PhaseRef: TypeAlias = str | Phase | TerminateTarget | None


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
    state_vars: tuple[str, ...]
    issues: tuple[CompileIssue, ...]
    warnings: tuple[CompileIssue, ...] = ()
    phase_schedules: dict[str, tuple[str, ...]] = field(default_factory=dict)
    phase_dependency_edges: dict[str, tuple[tuple[str, str], ...]] = field(default_factory=dict)
    state_vars_without_initial: tuple[str, ...] = ()
    required_initial_state_vars: dict[str, tuple[str, ...]] = field(default_factory=dict)

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
    def minimal_initial_state_vars(self) -> tuple[str, ...]:
        return tuple(sorted(self.required_initial_state_vars))

    def format(self) -> str:
        lines = [
            f"ok = {self.ok}",
            f"issues = {_format_issue_list(self.issues)}",
            f"warnings = {_format_issue_list(self.warnings)}",
            f"minimal_initial_state_vars = {self.minimal_initial_state_vars}",
            f"state_vars_without_initial = {self.state_vars_without_initial}",
            f"required_initial_state_vars = {self.required_initial_state_vars}",
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


def _node_output_path(node: Node, output: VarPort[Any]) -> str:
    if output.name is None:
        return "<unbound>"
    return f"{node.node_id}.{output.name}"


def _node_input_path(node: Node, input_port: InputPort[Any]) -> str:
    if input_port.name is None:
        return "<unbound>"
    return f"{node.node_id}.{input_port.name}"


def _source_path(source: VarSource[Any]) -> str:
    source = _resolve_lazy_source(source)
    if isinstance(source, BoundVarPort):
        return source.path
    if isinstance(source, VarPort):
        return source.path
    if isinstance(source, SystemSource):
        return source.path
    if not isinstance(source, str):
        raise TypeError(
            "State source must be a VarPort, SystemSource, a string reference, "
            "or a zero-argument callable returning one."
        )
    return _normalize_output_path(source)


def _resolve_lazy_source(source: VarSource[T]) -> ResolvedVarSource[T]:
    if isinstance(source, (BoundVarPort, VarPort, SystemSource, str)):
        return source
    return source()


def _normalize_output_path(path: str) -> str:
    parts = path.split(".")
    if len(parts) == 3:
        return f"{parts[0]}.{parts[2]}"
    return path


def _z3_variable_for_type(output_type: type[Any], name: str) -> Any:
    if output_type is bool:
        return z3.Bool(name)
    if output_type is int:
        return z3.Int(name)
    if output_type is float:
        return z3.Real(name)
    if _is_enum_type(output_type):
        return z3.Int(name)
    raise TypeError(f"State variable type {output_type!r} is not supported by z3.")


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
