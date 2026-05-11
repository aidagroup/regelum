from __future__ import annotations

import inspect
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast, get_type_hints

import casadi as ca
from scipy.integrate import solve_ivp

from regelum.core import (
    _MISSING,
    BoundVarPort,
    InputPort,
    Node,
    NodeInputs,
    NodeState,
    StateSnapshot,
    SystemSource,
    VarPort,
    _parse_time_step,
)

Scalar = int | float
StateValue = Scalar | tuple["StateValue", ...] | list["StateValue"]


class CasadiTraceError(RuntimeError):
    pass


class ODENode(Node):
    _allow_schedule_dt = False
    State = NodeState

    def __init_subclass__(cls) -> None:
        if "dt" in cls.__dict__:
            raise TypeError(f"{cls.__name__} cannot define dt; set dt on ODESystem instead.")
        super().__init_subclass__()
        _install_dstate_input_ports(cls)
        cls._state_namespace_name = cls._output_namespace_name
        cls._state_namespace_cls = cls._output_namespace_cls
        cls._state_vars = cls._outputs

    def _bind_ports(self) -> None:
        super()._bind_ports()
        self._ode_state_values = {
            name: port.initial_value(self) for name, port in self.__class__._state_vars.items()
        }

    def state(self) -> NodeState:
        if self.__class__._state_namespace_cls is None:
            raise RuntimeError(f"{self.__class__.__name__} has no State namespace.")
        return self.__class__._state_namespace_cls(**self._ode_state_values)

    def dstate(self, inputs: Any, state: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def update(self, _inputs: NodeInputs | None = None) -> dict[str, Any]:
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

    def update(
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
            dstate = _call_dstate(node, inputs=inputs, state=state, time=time)
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


def _install_dstate_input_ports(node_cls: type[ODENode]) -> None:
    dstate_inputs = _collect_dstate_input_ports(node_cls)
    if not dstate_inputs:
        return
    if node_cls._inputs:
        node_cls._input_declaration_error = (
            "define ODE inputs either as a NodeInputs namespace or as dstate(...) "
            "Input parameters, not both"
        )
        return
    inputs_cls = type("Inputs", (NodeInputs,), dict(dstate_inputs))
    for name, port in dstate_inputs.items():
        port.__set_name__(inputs_cls, name)
        port.node_cls = node_cls
    node_cls.Inputs = inputs_cls  # ty: ignore[invalid-assignment]
    node_cls._input_namespace_name = "Inputs"
    node_cls._input_namespace_cls = inputs_cls
    node_cls._run_input_mode = "object"
    node_cls._inputs = dstate_inputs


def _collect_dstate_input_ports(node_cls: type[ODENode]) -> dict[str, InputPort[Any]]:
    dstate = node_cls.__dict__.get("dstate")
    if dstate is None:
        return {}
    try:
        signature = inspect.signature(dstate)
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


def _call_dstate(
    node: ODENode,
    *,
    inputs: NodeInputs,
    state: NodeState,
    time: Any,
) -> NodeState:
    try:
        signature = inspect.signature(node.dstate)
    except (TypeError, ValueError):
        return node.dstate(inputs, state)

    values = {
        "inputs": inputs,
        "state": state,
        "time": time,
    }
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    type_hints = _dstate_type_hints(node)

    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            raise TypeError("ODENode.dstate does not support *args parameters.")
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        value = _dstate_argument_value(node, parameter, type_hints, values)
        if value is _MISSING:
            if parameter.default is inspect.Parameter.empty:
                raise TypeError(
                    "ODENode.dstate parameters must be named inputs, state, or time; "
                    "annotated as the node's Inputs/State namespace; or declared as "
                    f"a dstate Input parameter; got required parameter {parameter.name!r}."
                )
            if isinstance(parameter.default, InputPort):
                raise TypeError(
                    f"ODENode.dstate Input parameter {parameter.name!r} was not registered."
                )
            continue
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            args.append(value)
        elif parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs[parameter.name] = value

    return node.dstate(*args, **kwargs)


def _dstate_argument_value(
    node: ODENode,
    parameter: inspect.Parameter,
    type_hints: dict[str, Any],
    values: dict[str, Any],
) -> Any:
    if parameter.name == "time":
        return values["time"]
    if parameter.name in ("inputs", "state"):
        return values[parameter.name]
    if parameter.name in node.__class__._inputs:
        return getattr(values["inputs"], parameter.name)
    if isinstance(parameter.default, InputPort):
        return getattr(values["inputs"], parameter.name, _MISSING)
    annotation = type_hints.get(parameter.name, parameter.annotation)
    if _is_inputs_annotation(node, annotation):
        return values["inputs"]
    if _is_state_annotation(node, annotation):
        return values["state"]
    return _MISSING


def _dstate_type_hints(node: ODENode) -> dict[str, Any]:
    module = sys.modules.get(node.__class__.__module__)
    globalns = vars(module) if module is not None else {}
    localns: dict[str, Any] = {node.__class__.__name__: node.__class__}
    input_namespace_cls = node.__class__._input_namespace_cls
    state_namespace_name = node.__class__._state_namespace_name
    state_namespace_cls = node.__class__._state_namespace_cls
    if input_namespace_cls is not None:
        localns["Inputs"] = input_namespace_cls
        if node.__class__._input_namespace_name is not None:
            localns[node.__class__._input_namespace_name] = input_namespace_cls
    if state_namespace_cls is not None:
        localns["State"] = state_namespace_cls
        if state_namespace_name is not None:
            localns[state_namespace_name] = state_namespace_cls
    try:
        return get_type_hints(node.dstate, globalns=globalns, localns=localns)
    except Exception:
        return {}


def _is_inputs_annotation(node: ODENode, annotation: Any) -> bool:
    input_namespace_cls = node.__class__._input_namespace_cls
    if input_namespace_cls is not None and _is_subclass(annotation, input_namespace_cls):
        return True
    return _annotation_name(annotation) in {"Inputs", node.__class__._input_namespace_name}


def _is_state_annotation(node: ODENode, annotation: Any) -> bool:
    state_namespace_cls = node.__class__._state_namespace_cls
    if state_namespace_cls is not None and _is_subclass(annotation, state_namespace_cls):
        return True
    return _annotation_name(annotation) in {"State", node.__class__._state_namespace_name}


def _is_subclass(annotation: Any, parent: type[Any]) -> bool:
    try:
        return isinstance(annotation, type) and issubclass(annotation, parent)
    except TypeError:
        return False


def _annotation_name(annotation: Any) -> str | None:
    if isinstance(annotation, str):
        return annotation
    return getattr(annotation, "__name__", None)


def _source_path(source: Any) -> str:
    if callable(source) and not isinstance(source, VarPort):
        source = source()
    if isinstance(source, BoundVarPort):
        return source.path
    if isinstance(source, VarPort):
        return source.path
    if isinstance(source, SystemSource):
        return source.path
    if not isinstance(source, str):
        raise TypeError(
            "ODE input source must be an VarPort, a string reference, "
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
