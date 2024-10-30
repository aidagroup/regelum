"""This module contains the base class for all transistors that compute transitions of nodes composing an environment."""

from __future__ import annotations
from typing import Callable, Optional, Dict, Any, List
from abc import abstractmethod, ABC
from regelum.environment.node import Node, State, Inputs
from casadi import integrator, MX, vertcat, vec, DM
import numpy as np
from scipy.integrate import solve_ivp


def register_transition(*paths):
    def decorator(method):
        if not hasattr(method, "transition_paths"):
            method.transition_paths = []
        method.transition_paths.extend(paths)
        return method

    return decorator


class Transistor:
    """An entity representing a state transition of a node in the environment."""

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
            # Use registered transition methods
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


class ODETransistor(Transistor):
    """A base class for computing the evolution of the state of a node represented as an ODE."""

    class IntegratorInterface(ABC):
        """An interface for an ODE integrator."""

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
    """An ODETransistor that uses CasADi to compute the state transitions."""

    class CasADiIntegrator(ODETransistor.IntegratorInterface):
        """An ODE integrator that uses CasADi for state transitions."""

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
    """ODETransistor using scipy's solve_ivp for numerical integration."""

    class ScipyIntegrator(ODETransistor.IntegratorInterface):
        def create(self):
            def wrapped_dynamics(t, x, *args):
                # Convert args tuple to numpy array for reshaping
                args = np.array(args)

                # Reshape inputs for the dynamics function
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

                # Call dynamics and return flattened result
                dynamics_result = self.state_dynamics_function()
                return np.array(list(dynamics_result.values())[0]).flatten()

            return wrapped_dynamics

    def setup_integrator(self):
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
