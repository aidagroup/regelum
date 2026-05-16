from __future__ import annotations

import inspect
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Literal, cast, get_type_hints

import casadi as ca
import numpy as np
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
StateValue = Scalar | tuple["StateValue", ...] | list["StateValue"] | np.ndarray[Any, Any]
OdeBackend = Literal["scipy", "casadi"]


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
        return cast(NodeState, self.__class__._state_namespace_cls(**self._ode_state_values))

    def dstate(self, inputs: Any, state: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def update(self, _inputs: NodeInputs | None = None) -> dict[str, Any]:
        return dict(self._ode_state_values)


@dataclass(frozen=True)
class _StateField:
    node: ODENode
    name: str
    shape: "_ShapeSpec"
    size: int


@dataclass(frozen=True)
class _ParameterField:
    path: str
    shape: "_ShapeSpec"
    size: int


@dataclass
class _CasadiGraph:
    parameter_fields: tuple[_ParameterField, ...]
    rhs_function: ca.Function
    jacobian_function: ca.Function
    integrator_dae: dict[str, Any]
    integrators: dict[float, ca.Function] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class _ShapeSpec:
    kind: Literal["scalar", "list", "tuple", "ndarray"]
    dims: tuple[int, ...]

    @property
    def size(self) -> int:
        if not self.dims:
            return 1
        size = 1
        for dim in self.dims:
            size *= dim
        return size


class ODESystem(Node):
    _is_ode_system = True

    def __init__(
        self,
        *,
        nodes: Sequence[ODENode],
        dt: Any,
        method: str = "LSODA",
        backend: OdeBackend = "scipy",
        options: dict[str, Any] | None = None,
    ) -> None:
        if backend not in ("scipy", "casadi"):
            raise ValueError("ODESystem.backend must be 'scipy' or 'casadi'.")
        if backend == "casadi" and method == "LSODA":
            raise ValueError(
                "ODESystem backend='casadi' cannot use method='LSODA'; "
                "use a CasADi integrator plugin such as method='cvodes'."
            )
        self.nodes = tuple(nodes)
        self.dt = _parse_time_step(dt, field_name="ODESystem.dt")
        self.method = method
        self.backend = backend
        self.options = dict(options or {})
        if self.backend == "casadi":
            reserved = {"t0", "tf", "grid", "output_t0"} & set(self.options)
            if reserved:
                raise ValueError(
                    "CasADi ODESystem options must not define integration time "
                    f"options {sorted(reserved)}; Regelum controls time via Clock.time."
                )
        self._time_s = 0.0
        self._casadi_graph: _CasadiGraph | None = None
        self._fields = tuple(
            _StateField(
                node=node,
                name=name,
                shape=_shape(value, field_name=f"{node.node_id}.{name}"),
                size=len(_flatten(value, field_name=f"{node.node_id}.{name}")),
            )
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
        if time_start is None or time_stop is None:
            raise ValueError(
                "ODESystem.update requires time_start and time_stop; "
                "run ODESystem through PhasedReactiveSystem or pass the global time interval."
            )
        x0 = self._pack_state()
        graph = self._get_casadi_graph(state_snapshot)
        parameters = self._pack_parameters(graph.parameter_fields, state_snapshot)
        start = float(time_start)
        stop = float(time_stop)
        if self.backend == "scipy":
            result = solve_ivp(
                lambda time_s, state: _casadi_vector(
                    graph.rhs_function(time_s, state, parameters)
                ),
                (start, stop),
                x0,
                method=self.method,
                jac=lambda _time_s, state: _casadi_matrix(
                    graph.jacobian_function(_time_s, state, parameters)
                ),
                **self.options,
            )
            self._unpack_state([float(value) for value in result.y[:, -1]])
        else:
            integrator = self._get_casadi_integrator(graph, stop - start)
            result = integrator(x0=x0, p=[start, *parameters])
            self._unpack_state(_casadi_vector(result["xf"]))
        self._time_s = stop
        return {}

    def _get_casadi_graph(self, state_snapshot: StateSnapshot) -> _CasadiGraph:
        if self._casadi_graph is None:
            self._casadi_graph = self._build_casadi_graph(state_snapshot)
        return self._casadi_graph

    def _pack_state(self) -> list[float]:
        values: list[float] = []
        for field in self._fields:
            values.extend(
                float(value)
                for value in _flatten(
                    field.node._ode_state_values[field.name],
                    field_name=f"{field.node.node_id}.{field.name}",
                )
            )
        return values

    def _unpack_state(self, values: Sequence[float]) -> None:
        offset = 0
        for field in self._fields:
            chunk = values[offset : offset + field.size]
            field.node._ode_state_values[field.name] = _restore_value(chunk, field.shape)
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

        derivatives: list[Any] = []
        fields_by_node = {
            node: tuple(field for field in self._fields if field.node is node) for node in self.nodes
        }
        for node in self.nodes:
            inputs = _build_node_inputs(node, snapshot)
            state = node.state()
            dstate = _call_dstate(node, inputs=inputs, state=state, time=time)
            for field in fields_by_node[node]:
                flattened = _flatten(
                    getattr(dstate, field.name),
                    field_name=f"{node.node_id}.{field.name}",
                    allow_bool=False,
                )
                if len(flattened) != field.size:
                    raise ValueError(
                        f"{node.__class__.__name__}.dstate returned {len(flattened)} values "
                        f"for {field.name!r}, expected {field.size} from init shape."
                    )
                derivatives.extend(flattened)
        return derivatives

    def _trace_dstate(
        self,
        *,
        state_snapshot: StateSnapshot,
        parameter_fields: tuple[_ParameterField, ...],
        x: ca.MX,
        p: ca.MX,
        time: Any,
    ) -> ca.MX:
        parameter_snapshot = dict(state_snapshot)
        offset = 0
        for field in parameter_fields:
            chunk = [p[index] for index in range(offset, offset + field.size)]
            parameter_snapshot[field.path] = _casadi_view(chunk, field.shape)
            offset += field.size

        offset = 0
        for field in self._fields:
            chunk = [x[index] for index in range(offset, offset + field.size)]
            field.node._ode_state_values[field.name] = _casadi_view(chunk, field.shape)
            offset += field.size
        return ca.vertcat(*self._dstate_from_current_node_state(parameter_snapshot, time=time))

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
            dstate = self._trace_dstate(
                state_snapshot=state_snapshot,
                parameter_fields=parameter_fields,
                x=x,
                p=p,
                time=time,
            )
            integrator_p = ca.MX.sym("p", 1 + sum(field.size for field in parameter_fields))
            tau = ca.MX.sym("tau")
            integrator_dstate = self._trace_dstate(
                state_snapshot=state_snapshot,
                parameter_fields=parameter_fields,
                x=x,
                p=integrator_p[1:],
                time=integrator_p[0] + tau,
            )
            jacobian = ca.jacobian(dstate, x)
            return _CasadiGraph(
                parameter_fields=parameter_fields,
                rhs_function=ca.Function("ode_rhs", [time, x, p], [dstate]),
                jacobian_function=ca.Function("ode_jacobian", [time, x, p], [jacobian]),
                integrator_dae={
                    "x": x,
                    "p": integrator_p,
                    "t": tau,
                    "ode": integrator_dstate,
                },
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
                shape = _shape(value, field_name=path, allow_bool=True)
                fields.append(
                    _ParameterField(
                        path=path,
                        shape=shape,
                        size=len(_flatten(value, field_name=path, allow_bool=True)),
                    )
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
            value = state_snapshot[field.path]
            shape = _shape(value, field_name=field.path, allow_bool=True)
            if shape != field.shape:
                raise ValueError(
                    f"ODE input {field.path!r} changed shape from {field.shape.dims} "
                    f"to {shape.dims}; input shapes must stay fixed after graph build."
                )
            values.extend(
                float(value)
                for value in _flatten(value, field_name=field.path, allow_bool=True)
            )
        return values

    def _get_casadi_integrator(self, graph: _CasadiGraph, duration: float) -> ca.Function:
        if duration <= 0:
            raise ValueError("ODESystem integration interval must have positive duration.")
        try:
            return graph.integrators[duration]
        except KeyError:
            name = f"ode_integrator_{len(graph.integrators)}"
            integrator = ca.integrator(
                name,
                self.method,
                graph.integrator_dae,
                0.0,
                duration,
                self.options,
            )
            graph.integrators[duration] = integrator
            return integrator


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


def _flatten(value: Any, *, field_name: str, allow_bool: bool = False) -> list[Any]:
    if _is_numeric_scalar(value, allow_bool=allow_bool) or _is_casadi_scalar(value):
        return [value]
    if _is_casadi_value(value):
        rows, cols = cast(tuple[int, int], value.shape)
        return [value[row, col] for row in range(rows) for col in range(cols)]
    if isinstance(value, np.ndarray):
        _shape(value, field_name=field_name, allow_bool=allow_bool)
        return list(value.reshape((-1,)))
    if isinstance(value, (tuple, list)):
        _shape(value, field_name=field_name, allow_bool=allow_bool)
        flattened: list[Any] = []
        for item in value:
            flattened.extend(_flatten(item, field_name=field_name, allow_bool=allow_bool))
        return flattened
    raise TypeError(
        f"ODE values for {field_name} must be numeric scalars, list/tuple arrays, "
        f"numpy arrays, or CasADi values; got {value!r}."
    )


def _shape(value: Any, *, field_name: str, allow_bool: bool = False) -> _ShapeSpec:
    if _is_numeric_scalar(value, allow_bool=allow_bool) or _is_casadi_scalar(value):
        return _ShapeSpec("scalar", ())
    if _is_casadi_value(value):
        rows, cols = cast(tuple[int, int], value.shape)
        if rows == 0 or cols == 0:
            raise ValueError(f"ODE value {field_name} must not be empty.")
        if cols == 1:
            return _ShapeSpec("ndarray", (rows,))
        return _ShapeSpec("ndarray", (rows, cols))
    if isinstance(value, np.ndarray):
        if value.size == 0:
            raise ValueError(f"ODE value {field_name} must not be an empty array.")
        if value.ndim > 2:
            raise ValueError(f"ODE value {field_name} must be scalar, 1D, or 2D; got {value.ndim}D.")
        for item in value.reshape((-1,)):
            if not _is_numeric_scalar(item, allow_bool=allow_bool):
                raise TypeError(f"ODE array {field_name} must contain only numeric values.")
        return _ShapeSpec("ndarray", tuple(int(dim) for dim in value.shape))
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError(f"ODE value {field_name} must not be an empty array.")
        dims = _sequence_dims(value, field_name=field_name, allow_bool=allow_bool, depth=1)
        return _ShapeSpec("tuple" if isinstance(value, tuple) else "list", dims)
    raise TypeError(
        f"ODE values for {field_name} must be numeric scalars, list/tuple arrays, "
        f"numpy arrays, or CasADi values; got {value!r}."
    )


def _sequence_dims(
    value: Sequence[Any],
    *,
    field_name: str,
    allow_bool: bool,
    depth: int,
) -> tuple[int, ...]:
    if not value:
        raise ValueError(f"ODE value {field_name} must not contain empty arrays.")
    nested = [isinstance(item, (tuple, list)) for item in value]
    if any(nested):
        if not all(nested):
            raise ValueError(f"ODE value {field_name} must not mix scalars and arrays.")
        if depth >= 2:
            raise ValueError(f"ODE value {field_name} must be scalar, 1D, or 2D.")
        child_dims = [
            _sequence_dims(
                cast(Sequence[Any], item),
                field_name=field_name,
                allow_bool=allow_bool,
                depth=depth + 1,
            )
            for item in value
        ]
        first = child_dims[0]
        if any(dims != first for dims in child_dims):
            raise ValueError(f"ODE value {field_name} must be rectangular, not ragged.")
        return (len(value), *first)
    for item in value:
        if not (_is_numeric_scalar(item, allow_bool=allow_bool) or _is_casadi_scalar(item)):
            raise TypeError(f"ODE array {field_name} must contain only numeric values.")
    return (len(value),)


def _restore_value(values: Sequence[float], shape: _ShapeSpec) -> Any:
    if len(values) != shape.size:
        raise ValueError("Wrong number of values while unpacking ODE state.")
    floats = [float(value) for value in values]
    if shape.kind == "scalar":
        return floats[0]
    if shape.kind == "ndarray":
        return np.asarray(floats, dtype=float).reshape(shape.dims)
    if len(shape.dims) == 1:
        if shape.kind == "list":
            return list(floats)
        if shape.kind == "tuple":
            return tuple(floats)
    if len(shape.dims) == 2:
        rows, cols = shape.dims
        matrix = [floats[row * cols : (row + 1) * cols] for row in range(rows)]
        if shape.kind == "list":
            return matrix
        if shape.kind == "tuple":
            return tuple(tuple(row) for row in matrix)
    raise ValueError(f"Unknown or unsupported ODE shape: {shape!r}.")


def _casadi_view(values: Sequence[Any], shape: _ShapeSpec) -> Any:
    if len(values) != shape.size:
        raise ValueError("Wrong number of values while building CasADi ODE value.")
    if shape.kind == "scalar":
        return values[0]
    if len(shape.dims) == 1:
        return ca.vertcat(*values)
    if len(shape.dims) == 2:
        rows, cols = shape.dims
        return ca.vertcat(
            *[
                ca.horzcat(*values[row * cols : (row + 1) * cols])
                for row in range(rows)
            ]
        )
    raise ValueError(f"Unsupported CasADi ODE shape: {shape!r}.")


def _is_numeric_scalar(value: Any, *, allow_bool: bool) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return allow_bool
    return isinstance(value, (int, float, np.integer, np.floating))


def _is_casadi_scalar(value: Any) -> bool:
    return _is_casadi_value(value) and value.shape == (1, 1)


def _casadi_vector(value: ca.DM) -> list[float]:
    return [float(item) for item in value.full().reshape((-1,))]


def _casadi_matrix(value: ca.DM) -> list[list[float]]:
    return [[float(item) for item in row] for row in value.full().tolist()]


def _is_casadi_value(value: Any) -> bool:
    return isinstance(value, (ca.MX, ca.SX, ca.DM))
