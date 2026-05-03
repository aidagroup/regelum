from __future__ import annotations

import inspect
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, cast, dataclass_transform

import casadi as ca
from scipy.integrate import solve_ivp

from regelum.core import (
    _MISSING,
    BoundOutputPort,
    InitialValue,
    Node,
    NodeInputs,
    NodeOutputs,
    OutputPort,
    StateSnapshot,
    SystemSource,
    _parse_time_step,
)

Scalar = int | float
StateValue = Scalar | tuple["StateValue", ...] | list["StateValue"]


class CasadiTraceError(RuntimeError):
    pass


class StateVarPort(OutputPort[Any]):
    pass


def StateVar(
    *,
    initial: InitialValue = _MISSING,
    domain: Iterable[Any] | None = None,
) -> Any:
    return StateVarPort(initial=initial, domain=domain)


@dataclass_transform(field_specifiers=(StateVar,))
class NodeState:
    def __init__(self, **values: Any) -> None:
        for name, value in values.items():
            setattr(self, name, value)


class _BoundStateNamespace:
    def __init__(
        self,
        node: ODENode,
        nested_cls: type[NodeState],
        ports: dict[str, StateVarPort],
    ) -> None:
        self._node = node
        self._nested_cls = nested_cls
        self._ports = ports

    def __getattr__(self, name: str) -> BoundOutputPort[Any]:
        try:
            port = self._ports[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return BoundOutputPort(self._node, port)

    def __call__(self, **values: Any) -> NodeState:
        return self._nested_cls(**values)


class ODENode(Node):
    State = NodeState

    def __init_subclass__(cls) -> None:
        state_namespace_name, state_namespace_cls = _find_state_namespace(cls)
        state_vars = _collect_state_vars(state_namespace_cls)
        if state_vars:
            existing_outputs = _collect_existing_outputs(cls)
            cls.Outputs = type("Outputs", (NodeOutputs,), {**existing_outputs, **state_vars})
        super().__init_subclass__()
        cls._state_namespace_name = state_namespace_name
        cls._state_namespace_cls = state_namespace_cls
        cls._state_vars = state_vars

    def _bind_ports(self) -> None:
        super()._bind_ports()
        state_namespace_name = self.__class__._state_namespace_name
        state_namespace_cls = self.__class__._state_namespace_cls
        if state_namespace_name is not None and state_namespace_cls is not None:
            setattr(
                self,
                state_namespace_name,
                _BoundStateNamespace(self, state_namespace_cls, self.__class__._state_vars),
            )
        self._ode_state_values = {
            name: port.initial_value(self) for name, port in self.__class__._state_vars.items()
        }

    def state(self) -> NodeState:
        if self.__class__._state_namespace_cls is None:
            raise RuntimeError(f"{self.__class__.__name__} has no State namespace.")
        return self.__class__._state_namespace_cls(**self._ode_state_values)

    def dstate(self, inputs: Any, state: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def run(self, _inputs: NodeInputs | None = None) -> dict[str, Any]:
        return dict(self._ode_state_values)


@dataclass(frozen=True)
class _StateField:
    node: ODENode
    name: str
    shape: Any
    size: int


@dataclass(frozen=True)
class _ParameterField:
    path: str
    shape: Any
    size: int


@dataclass(frozen=True)
class _CasadiGraph:
    parameter_fields: tuple[_ParameterField, ...]
    rhs_function: ca.Function
    jacobian_function: ca.Function


class ODESystem(Node):
    _is_ode_system = True

    def __init__(
        self,
        *,
        nodes: Sequence[ODENode],
        dt: Any,
        method: str = "LSODA",
    ) -> None:
        self.nodes = tuple(nodes)
        self.dt = _parse_time_step(dt, field_name="ODESystem.dt")
        self.method = method
        self._time_s = 0.0
        self._casadi_graph: _CasadiGraph | None = None
        self._fields = tuple(
            _StateField(node=node, name=name, shape=_shape(value), size=len(_flatten(value)))
            for node in self.nodes
            for name, value in node._ode_state_values.items()
        )

    def run(
        self,
        _inputs: NodeInputs | None = None,
        *,
        state_snapshot: StateSnapshot,
        time_start: float | None = None,
        time_stop: float | None = None,
    ) -> dict[str, Any]:
        x0 = self._pack_state()
        graph = self._get_casadi_graph(state_snapshot)
        parameters = self._pack_parameters(graph.parameter_fields, state_snapshot)
        start = self._time_s if time_start is None else time_start
        stop = start + float(self.dt) if time_stop is None else time_stop
        result = solve_ivp(
            lambda time_s, state: _casadi_vector(graph.rhs_function(time_s, state, parameters)),
            (start, stop),
            x0,
            method=self.method,
            jac=lambda _time_s, state: _casadi_matrix(
                graph.jacobian_function(_time_s, state, parameters)
            ),
        )
        self._unpack_state([float(value) for value in result.y[:, -1]])
        self._time_s = stop
        return {}

    def _get_casadi_graph(self, state_snapshot: StateSnapshot) -> _CasadiGraph:
        if self._casadi_graph is None:
            self._casadi_graph = self._build_casadi_graph(state_snapshot)
        return self._casadi_graph

    def _pack_state(self) -> list[float]:
        values: list[float] = []
        for field in self._fields:
            values.extend(_flatten(field.node._ode_state_values[field.name]))
        return values

    def _unpack_state(self, values: Sequence[float]) -> None:
        offset = 0
        for field in self._fields:
            chunk = values[offset : offset + field.size]
            field.node._ode_state_values[field.name] = _unflatten(chunk, field.shape)
            offset += field.size

    def _dstate_from_current_node_state(
        self,
        state_snapshot: StateSnapshot,
        *,
        time: Any = None,
    ) -> list[Any]:
        snapshot = dict(state_snapshot)
        for node in self.nodes:
            for name, value in node._ode_state_values.items():
                snapshot[f"{node.node_id}.{name}"] = value
                snapshot[f"{node.__class__.__name__}.{name}"] = value

        derivatives: list[float] = []
        for node in self.nodes:
            inputs = _build_node_inputs(node, snapshot)
            state = node.state()
            if _accepts_dstate_time(node):
                dstate = node.dstate(inputs, state, time=time)
            else:
                dstate = node.dstate(inputs, state)
            derivatives.extend(
                value
                for name in node.__class__._state_vars
                for value in _flatten(getattr(dstate, name))
            )
        return derivatives

    def _build_casadi_graph(
        self,
        state_snapshot: StateSnapshot,
    ) -> _CasadiGraph:
        original_state = {node: dict(node._ode_state_values) for node in self.nodes}
        try:
            time = ca.MX.sym("t")
            x = ca.MX.sym("x", len(self._pack_state()))
            parameter_fields = self._parameter_fields(state_snapshot)
            p = ca.MX.sym("p", sum(field.size for field in parameter_fields))
            parameter_snapshot = dict(state_snapshot)
            offset = 0
            for field in parameter_fields:
                chunk = [p[index] for index in range(offset, offset + field.size)]
                parameter_snapshot[field.path] = _unflatten(chunk, field.shape)
                offset += field.size

            offset = 0
            for field in self._fields:
                chunk = [x[index] for index in range(offset, offset + field.size)]
                field.node._ode_state_values[field.name] = _unflatten(chunk, field.shape)
                offset += field.size
            dstate = ca.vertcat(
                *self._dstate_from_current_node_state(parameter_snapshot, time=time)
            )
            jacobian = ca.jacobian(dstate, x)
            return _CasadiGraph(
                parameter_fields=parameter_fields,
                rhs_function=ca.Function("ode_rhs", [time, x, p], [dstate]),
                jacobian_function=ca.Function("ode_jacobian", [time, x, p], [jacobian]),
            )
        except Exception as exc:
            raise CasadiTraceError(
                "CasADi backend cannot trace this ODESystem. "
                "Keep ODENode.dstate methods CasADi-traceable: use casadi primitives "
                "directly, such as ca.if_else/ca.sin/ca.cos/ca.sqrt, instead of Python "
                "if, math, or numpy operations over state/input values. State shapes must "
                "be fixed."
            ) from exc
        finally:
            for node, values in original_state.items():
                node._ode_state_values = values

    def _parameter_fields(self, state_snapshot: StateSnapshot) -> tuple[_ParameterField, ...]:
        state_paths = {
            f"{node.node_id}.{name}" for node in self.nodes for name in node.__class__._state_vars
        } | {
            f"{node.__class__.__name__}.{name}"
            for node in self.nodes
            for name in node.__class__._state_vars
        }
        fields: list[_ParameterField] = []
        seen: set[str] = set()
        for node in self.nodes:
            for input_port in node.__class__._inputs.values():
                source = input_port.source
                if source is None:
                    continue
                path = _source_path(source)
                if path in seen or path in state_paths:
                    continue
                if path not in state_snapshot:
                    raise KeyError(f"Cannot build CasADi ODE graph: missing input source {path!r}.")
                value = state_snapshot[path]
                fields.append(
                    _ParameterField(path=path, shape=_shape(value), size=len(_flatten(value)))
                )
                seen.add(path)
        return tuple(fields)

    def _pack_parameters(
        self,
        fields: tuple[_ParameterField, ...],
        state_snapshot: StateSnapshot,
    ) -> list[float]:
        values: list[float] = []
        for field in fields:
            values.extend(float(value) for value in _flatten(state_snapshot[field.path]))
        return values


def _find_state_namespace(node_cls: type[ODENode]) -> tuple[str | None, type[NodeState] | None]:
    own_namespaces = [
        (name, value)
        for name, value in node_cls.__dict__.items()
        if isinstance(value, type) and issubclass(value, NodeState) and value is not NodeState
    ]
    if len(own_namespaces) > 1:
        names = ", ".join(name for name, _ in own_namespaces)
        raise TypeError(
            f"{node_cls.__name__} may define zero or one state namespace; "
            f"found {len(own_namespaces)}: {names}."
        )
    if own_namespaces:
        return own_namespaces[0]
    return None, None


def _collect_state_vars(nested_cls: type[NodeState] | None) -> dict[str, StateVarPort]:
    if nested_cls is None:
        return {}
    return {
        name: value for name, value in vars(nested_cls).items() if isinstance(value, StateVarPort)
    }


def _collect_existing_outputs(node_cls: type[ODENode]) -> dict[str, OutputPort[Any]]:
    outputs = node_cls.__dict__.get("Outputs")
    if not isinstance(outputs, type) or not issubclass(outputs, NodeOutputs):
        return {}
    return {name: value for name, value in vars(outputs).items() if isinstance(value, OutputPort)}


def _build_node_inputs(node: ODENode, snapshot: StateSnapshot) -> NodeInputs:
    values: dict[str, Any] = {}
    for name, input_port in node.__class__._inputs.items():
        source = input_port.source
        if source is None:
            values[name] = input_port.default
            continue
        path = _source_path(source)
        values[name] = snapshot.get(path, input_port.default)
    return node.__class__.Inputs(**values)


def _accepts_dstate_time(node: ODENode) -> bool:
    try:
        signature = inspect.signature(node.dstate)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == "time" and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def _source_path(source: Any) -> str:
    if callable(source) and not isinstance(source, OutputPort):
        source = source()
    if isinstance(source, BoundOutputPort):
        return source.path
    if isinstance(source, OutputPort):
        return source.path
    if isinstance(source, SystemSource):
        return source.path
    if not isinstance(source, str):
        raise TypeError(
            "ODE input source must be an OutputPort, a string reference, "
            "or a zero-argument callable returning one."
        )
    return _normalize_path(source)


def _normalize_path(path: str) -> str:
    parts = path.split(".")
    if len(parts) == 3:
        return f"{parts[0]}.{parts[2]}"
    return path


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, (int, float)) or _is_casadi_value(value):
        return [value]
    if isinstance(value, (tuple, list)):
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten(item))
        return flattened
    raise TypeError(f"ODE state values must be numeric scalars or tuples/lists, got {value!r}.")


def _shape(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return None
    if isinstance(value, tuple):
        return ("tuple", tuple(_shape(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_shape(item) for item in value))
    raise TypeError(f"ODE state values must be numeric scalars or tuples/lists, got {value!r}.")


def _unflatten(values: Sequence[float], shape: Any) -> Any:
    value, used = _unflatten_at(values, shape, 0)
    if used != len(values):
        raise ValueError("Unused values while unpacking ODE state.")
    return value


def _unflatten_at(values: Sequence[float], shape: Any, offset: int) -> tuple[Any, int]:
    if shape is None:
        return values[offset], offset + 1
    kind, children = cast(tuple[str, tuple[Any, ...]], shape)
    items = []
    cursor = offset
    for child in children:
        value, cursor = _unflatten_at(values, child, cursor)
        items.append(value)
    if kind == "tuple":
        return tuple(items), cursor
    if kind == "list":
        return items, cursor
    raise ValueError(f"Unknown ODE state shape kind: {kind!r}.")


def _casadi_vector(value: ca.DM) -> list[float]:
    return [float(item) for item in value.full().reshape((-1,))]


def _casadi_matrix(value: ca.DM) -> list[list[float]]:
    return [[float(item) for item in row] for row in value.full().tolist()]


def _is_casadi_value(value: Any) -> bool:
    return isinstance(value, (ca.MX, ca.SX, ca.DM))
