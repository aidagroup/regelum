"""Transistor module for computing state transitions in environment nodes.

This module provides base classes for implementing different types of state transitions:
    - Transistor: Base class for state transitions
    - ODETransistor: Base class for ODE-based transitions
    - CasADiTransistor: CasADi-based ODE solver
    - ScipyTransistor: SciPy-based ODE solver
    - TransistorFactory: Factory for creating modified transistors
    - SampleAndHoldFactory: Factory for sample-and-hold behavior
"""

from __future__ import annotations
from typing import Callable, Optional, Dict, Any, List, TYPE_CHECKING
from abc import abstractmethod, ABC
from casadi import integrator, MX, vertcat, vec, DM
import numpy as np
from scipy.integrate import solve_ivp

if TYPE_CHECKING:
    from regelum.environment.node import Node, State, Inputs


def register_transition(*paths):
    def decorator(method):
        if not hasattr(method, "transition_paths"):
            method.transition_paths = []
        method.transition_paths.extend(paths)
        return method

    return decorator


class Transistor:
    """Base transistor for computing state transitions.

    Args:
        node: Node to compute transitions for
        time_final: Optional end time for simulation
    """

    def __init__(
        self,
        node: Node,
        time_final: Optional[float] = None,
    ) -> None:
        """Instantiate a Transistor with the associated Node."""
        self.node = node
        self.step_size = node.step_size
        self.time_final = time_final
        self.current_state = node.state
        self.transition_map = {}
        self._collect_transition_methods()
        self._create_default_transition()

    def _collect_transition_methods(self):
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, "transition_paths"):
                for path in attr.transition_paths:
                    self.transition_map[path] = attr

    def _create_default_transition(self):
        if not self.transition_map and hasattr(self.node, "compute_state_dynamics"):
            self.transition_map["default"] = self.node.compute_state_dynamics

    def step(self):
        inputs = self.collect_inputs()
        if any(state.value["value"] is None for state in self.node.inputs.states):
            return
        state_updates = {}

        if "default" in self.transition_map:
            new_state_values = self.transition_map["default"]()
            state_updates.update(new_state_values)
        else:
            for path, method in self.transition_map.items():
                state_to_update = self.current_state.search_by_path(path)
                if state_to_update is None:
                    raise ValueError(f"State path '{path}' not found.")
                new_value = method(state_to_update, inputs)
                state_updates[path] = new_value

        self.update_state(state_updates)

    def update_state(self, state_updates):
        for path, value in state_updates.items():
            state_to_update = self.current_state.search_by_path(path)
            if state_to_update:
                state_to_update.value = value
            else:
                raise ValueError(f"State path '{path}' not found during update.")

    def collect_inputs(self):
        return {state.name: state.value for state in self.node.inputs.states}

    @classmethod
    def with_modifier(cls, modifier):
        class ModifiedTransistor(cls):
            def __init__(self, node, **kwargs):
                super().__init__(node, **kwargs)
                modifier.node = self.node  # Bind the node to the modifier
                self.transition_map = modifier.transition_modifier(self.transition_map)

        return ModifiedTransistor


class ODETransistor(Transistor):
    """Base class for ODE-based state transitions.

    Args:
        node: Node to compute transitions for
        time_final: Optional end time for simulation
        time_start: Initial time
        dynamic_variable_paths: Paths to dynamic state variables
    """

    class IntegratorInterface(ABC):
        """Interface for ODE integrators.

        Args:
            state: State to integrate
            inputs: Input dependencies
            step_size: Integration step size
            state_dynamics_function: System dynamics function
        """

        def __init__(
            self,
            state: State,
            inputs: Inputs,
            step_size: float,
            state_dynamics_function: Callable,
        ) -> None:
            """Instantiate an ODE integrator."""
            self.time = 0.0
            self.state_info = state
            self.inputs_info = inputs
            self.step_size = step_size
            self.state_dynamics_function = state_dynamics_function

        @abstractmethod
        def create(self):
            """Create the integrator instance."""
            pass

    def __init__(
        self,
        node: Node,
        time_final: Optional[float] = None,
        time_start: float = 0.0,
        dynamic_variable_paths: Optional[List[str]] = None,
    ) -> None:
        """Instantiate an ODETransistor."""
        super().__init__(node=node, time_final=time_final)
        self.time = self.time_start = time_start
        self.dynamic_variable_paths = dynamic_variable_paths
        self.integrator = None
        self.setup_integrator()

    @abstractmethod
    def setup_integrator(self):
        """Set up the ODE integrator."""
        pass

    def _create_default_transition(self):
        """Create the default transition using the integrator."""
        if not self.transition_map:
            self.transition_map["default"] = self.ode_transition

    @abstractmethod
    def ode_transition(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute the new state using the ODE integrator."""
        pass


class CasADiTransistor(ODETransistor):
    """CasADi-based ODE solver for state transitions.

    Uses CasADi's numerical integration tools for computing state evolution.
    """

    class CasADiIntegrator(ODETransistor.IntegratorInterface):
        """CasADi-specific ODE integrator implementation.

        Args:
            state: State to integrate
            inputs: Input dependencies
            step_size: Integration step size
            state_dynamics_function: System dynamics function
            dynamic_variable_paths: Paths to dynamic state variables
        """

        def __init__(
            self,
            state: State,
            inputs: Inputs,
            step_size: float,
            state_dynamics_function: Callable,
            dynamic_variable_paths: Optional[List[str]] = None,
        ) -> None:
            super().__init__(state, inputs, step_size, state_dynamics_function)
            self.dynamic_variable_paths = dynamic_variable_paths

        def create(self):
            import regelum as rg

            with rg.symbolic_inference():
                inputs_symbolic_vector = self._create_inputs_vector()
                state_vector = self._create_state_vector()
                state_dynamics = self.state_dynamics_function()

                return integrator(
                    "integrator",
                    "rk",
                    {
                        "x": state_vector,
                        "p": inputs_symbolic_vector,
                        "ode": list(state_dynamics.values())[0],
                    },
                    0,
                    self.step_size,
                )

        def _create_inputs_vector(self) -> MX:
            inputs_symbolic_dict = self.inputs_info.collect()
            return (
                vertcat(*[vec(k) for k in inputs_symbolic_dict.values()])
                if inputs_symbolic_dict
                else MX([])
            )

        def _create_state_vector(self) -> MX:
            if not self.dynamic_variable_paths:
                if isinstance(self.state_info.data, dict):
                    raise ValueError("Tree-like state requires dynamic_variable_paths")
                return self.state_info.data

            state_components = []
            for path in self.dynamic_variable_paths:
                state = self.state_info.search_by_path(path)
                if not state:
                    raise ValueError(
                        f"Dynamic variable path '{path}' not found in state"
                    )
                state_components.append(state.data)
            return vertcat(*[vec(k) for k in state_components])

    def setup_integrator(self):
        self.integrator = self.CasADiIntegrator(
            state=self.node.state,
            inputs=self.node.inputs,
            step_size=self.step_size,
            state_dynamics_function=self.node.compute_state_dynamics,
            dynamic_variable_paths=self.dynamic_variable_paths,
        ).create()

    def ode_transition(self) -> Dict[str, Any]:
        inputs = self.collect_inputs()
        x0 = self._prepare_initial_state()
        p = self._prepare_parameters(inputs)

        res = self.integrator(x0=x0, p=p)
        return self._process_results(res["xf"])

    def _prepare_initial_state(self) -> DM:
        if self.dynamic_variable_paths:
            return self._prepare_dynamic_state()
        return self._prepare_simple_state()

    def _prepare_dynamic_state(self) -> DM:
        state_components = []
        for path in self.dynamic_variable_paths:
            state = self.node.state.search_by_path(path)
            if not state:
                raise ValueError(f"Dynamic variable path '{path}' not found in state")
            state_val = (
                DM(state.data)
                if isinstance(state.data, np.ndarray)
                else DM([state.data])
            )
            state_components.append(state_val)
        return vertcat(*[vec(k) for k in state_components])

    def _prepare_simple_state(self) -> DM:
        x0 = self.node.state.data
        if isinstance(x0, dict):
            raise ValueError("Tree-like state requires dynamic_variable_paths")
        return DM(x0) if isinstance(x0, np.ndarray) else DM([x0])

    def _prepare_parameters(self, inputs: Dict[str, Any]) -> DM:
        input_values = [
            inputs[state.name]["value"] for state in self.node.inputs.states
        ]
        return vertcat(*[vec(DM(val)) for val in input_values])

    def _process_results(self, xf: DM) -> Dict[str, Any]:
        shapes = self.node.state.get_shapes()
        new_state_value = np.array(xf.full()).flatten()

        if not self.dynamic_variable_paths:
            return {self.node.state.name: new_state_value}

        result = {}
        start_idx = 0
        for path in self.dynamic_variable_paths:
            size = int(np.prod(shapes[path]))
            slice_value = new_state_value[start_idx : start_idx + size]
            result[path] = (
                slice_value.reshape(shapes[path])
                if len(shapes[path]) > 0
                else slice_value[0]
            )
            start_idx += size
        return result


class ScipyTransistor(ODETransistor):
    """SciPy-based ODE solver for state transitions.

    Uses SciPy's solve_ivp for numerical integration of system dynamics.
    """

    class ScipyIntegrator(ODETransistor.IntegratorInterface):
        """SciPy-specific ODE integrator implementation."""

        def create(self):
            def wrapped_dynamics(t, x, *args):
                args = np.array(args)
                inputs_dict = {}
                start_idx = 0
                for state in self.inputs_info.states:
                    shape = (
                        state.value["value"].shape
                        if hasattr(state.value["value"], "shape")
                        else (1,)
                    )
                    size = int(np.prod(shape))
                    inputs_dict[state.name] = {
                        "value": args[start_idx : start_idx + size].reshape(shape)
                    }
                    start_idx += size

                # Set the current state before computing dynamics
                self.state_info.data = x
                dynamics_result = self.state_dynamics_function()
                return np.array(list(dynamics_result.values())[0]).flatten()

            return wrapped_dynamics

    def setup_integrator(self):
        assert self.step_size is not None, "Step size must be set"
        self.integrator = self.ScipyIntegrator(
            state=self.node.state,
            inputs=self.node.inputs,
            step_size=self.step_size,
            state_dynamics_function=self.node.compute_state_dynamics,
        ).create()

    def ode_transition(self) -> Dict[str, Any]:
        inputs = self.collect_inputs()
        x0 = self._prepare_initial_state()
        p = self._prepare_parameters(inputs)

        solution = solve_ivp(
            fun=self.integrator,
            t_span=(0, self.step_size),
            y0=x0.flatten(),
            args=tuple(p.flatten()),
            method="RK45",
            t_eval=[self.step_size],
        )

        return self._process_results(solution.y[:, -1])

    def _prepare_initial_state(self) -> np.ndarray:
        if self.dynamic_variable_paths:
            return self._prepare_dynamic_state()
        return self._prepare_simple_state()

    def _prepare_dynamic_state(self) -> np.ndarray:
        state_components = []
        for path in self.dynamic_variable_paths:
            state = self.node.state.search_by_path(path)
            if not state:
                raise ValueError(f"Dynamic variable path '{path}' not found in state")
            state_val = np.array(state.data).flatten()
            state_components.append(state_val)
        return np.concatenate(state_components)

    def _prepare_simple_state(self) -> np.ndarray:
        x0 = self.node.state.data
        if isinstance(x0, dict):
            raise ValueError("Tree-like state requires dynamic_variable_paths")
        return np.array(x0).flatten()

    def _prepare_parameters(self, inputs: Dict[str, Any]) -> np.ndarray:
        input_values = [
            inputs[state.name]["value"] for state in self.node.inputs.states
        ]
        return np.concatenate([np.array(val).flatten() for val in input_values])

    def _process_results(self, xf: np.ndarray) -> Dict[str, Any]:
        shapes = self.node.state.get_shapes()
        new_state_value = xf.flatten()

        if not self.dynamic_variable_paths:
            return {self.node.state.name: new_state_value}

        result = {}
        start_idx = 0
        for path in self.dynamic_variable_paths:
            size = int(np.prod(shapes[path]))
            slice_value = new_state_value[start_idx : start_idx + size]
            result[path] = (
                slice_value.reshape(shapes[path])
                if len(shapes[path]) > 0
                else slice_value[0]
            )
            start_idx += size
        return result


class TransistorFactory:
    """Factory for creating modified transistors.

    Args:
        transition_modifier: Function to modify transition mappings
    """

    def __init__(
        self, transition_modifier: Callable[[Dict[str, Callable]], Dict[str, Callable]]
    ) -> None:
        self.transition_modifier = transition_modifier

    def create(self, transistor: Transistor) -> Transistor:
        new_transistor = type(transistor)(
            node=transistor.node, time_final=transistor.time_final
        )
        new_transistor.transition_map = self.transition_modifier(
            transistor.transition_map
        )
        return new_transistor


class SampleAndHoldFactory(TransistorFactory):
    """Factory for creating sample-and-hold behavior in transistors.

    Creates transistors that update state at fixed intervals and hold values between updates.
    """

    def __init__(self) -> None:
        super().__init__(self._create_sample_and_hold_transition)

    def _create_sample_and_hold_transition(
        self, transition_map: Dict[str, Callable]
    ) -> Dict[str, Callable]:
        original_transition = transition_map["default"]
        last_update_time = 0.0
        cached_state = None

        def sample_and_hold_transition(transistor_self) -> Dict[str, Any]:
            nonlocal last_update_time, cached_state
            inputs = transistor_self.node.inputs.collect()
            current_time = float(inputs["Clock"])

            if (
                cached_state is None
                or current_time >= last_update_time + transistor_self.node.step_size
            ):
                cached_state = original_transition()
                last_update_time = current_time

            return cached_state

        def bound_transition():
            return sample_and_hold_transition(self)

        return {"default": bound_transition}
