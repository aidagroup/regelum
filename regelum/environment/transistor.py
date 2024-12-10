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
from typing import Callable, Optional, Dict, Any, List, TYPE_CHECKING, Type
from abc import abstractmethod, ABC
from casadi import integrator, MX, vertcat, vec, DM
import numpy as np
from scipy.integrate import solve_ivp
from copy import deepcopy

if TYPE_CHECKING:
    from regelum.environment.node.base import Node, State, Inputs


def register_transition(*paths):
    def decorator(method):
        if not hasattr(method, "transition_paths"):
            method.transition_paths = []
        method.transition_paths.extend(paths)
        return method

    return decorator


class Transistor:
    """Base class for computing state transitions in a Node.

    The Transistor is responsible for advancing the state of a Node from one timestep
    to the next. It retrieves the Node's inputs, calls the Node's defined dynamics function,
    and applies the resulting updates in-place to the Node's states.

    By default, the base Transistor expects the Node to implement a `compute_state_dynamics()`
    method. This method should return a dictionary mapping full state paths to their new values.

    If you need more complex behavior (e.g., ODE integration, sample-and-hold, reset behavior),
    you can subclass Transistor or wrap it using a TransistorFactory-based approach.

    Typical Usage:
        - At each timestep:
          1. The Graph calls `transistor.step()`
          2. The transistor collects inputs from the Node's input states.
          3. It calls `node.compute_state_dynamics()` to get the next state values.
          4. It updates the Node's state fields in-place with the computed values.

    Requirements for the Node:
        - The Node must implement `compute_state_dynamics()` returning a dict of {full_path: new_value}.
        - All required input states must be resolved and present at Node.inputs.

    Exceptions:
        - Raises `NotImplementedError` if `compute_state_dynamics()` is not defined in the Node.
        - Raises `ValueError` if any required state path to update does not exist in the Node's state.

    Example:
        class MyNode(Node):
            state = State("my_state", (1,), np.zeros(1))
            inputs = Inputs(["some_input"])

            def compute_state_dynamics(self):
                x = self.state.data
                u = self.inputs["some_input"].data
                return {"my_state": x + u}

        node = MyNode()
        transistor = Transistor(node=node)
        transistor.step()  # Node state updated in-place based on compute_state_dynamics
    """

    def __init__(self, node: Node, time_final: Optional[float] = None) -> None:
        """Initialize the Transistor.

        Args:
            node: The Node whose states this transistor will update.
            time_final: Optional simulation end time for reference (not used in the base transistor).
        """
        self.node = node
        self.step_size = node.step_size
        self.time_final = time_final
        self.current_state = node.state

        if not hasattr(node, "compute_state_dynamics") or not callable(
            node.compute_state_dynamics
        ):
            raise NotImplementedError(
                f"The node {node.state.name} does not implement `compute_state_dynamics()` method."
            )

    def is_inputs_ready(self):
        return all(
            state.value["value"] is not None for state in self.node.inputs.states
        )

    def step(self):
        """Advance the Node's state by one timestep.

        This method:
          1. Collects inputs from the Node's input states.
          2. Calls node.compute_state_dynamics() to obtain the state updates.
          3. Applies these updates directly to the Node's states.

        If any input state is currently undefined, this step does nothing.
        """
        # Check all inputs are available
        if not self.is_inputs_ready():
            # If inputs are not ready, we do nothing. This can happen if some dependency isn't computed yet.
            return

        # Compute new state dynamics
        new_state_values = self.node.compute_state_dynamics()

        # Update states in-place
        self._apply_updates(new_state_values)

    def collect_inputs(self):
        return {state.name: state.value for state in self.node.inputs.states}

    def _apply_updates(self, updates: Dict[str, Any]):
        """Apply computed updates to the Node's states in-place.

        Args:
            updates: A dictionary {full_state_path: new_value}.

        Raises:
            ValueError: If a provided state path does not exist in the node's state.
        """
        for path, value in updates.items():
            state_to_update = self.current_state.search_by_path(path)
            if state_to_update is None:
                raise ValueError(
                    f"State path '{path}' not found in node '{self.node.state.name}' during state update."
                )
            state_to_update.value = value


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
            """Initialize the IntegratorInterface.

            Args:
                state: State to integrate.
                inputs: Input dependencies.
                step_size: Integration step size.
                state_dynamics_function: System dynamics function.
            """
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
        """Initialize the ODETransistor.

        Args:
            node: Node to compute transitions for.
            time_final: Optional end time for simulation.
            time_start: Initial time for simulation.
            dynamic_variable_paths: Paths to dynamic state variables.
        """
        super().__init__(node=node, time_final=time_final)
        self.time = self.time_start = time_start
        self.dynamic_variable_paths = dynamic_variable_paths
        self.integrator = None
        self.setup_integrator()

    @abstractmethod
    def setup_integrator(self):
        """Set up the ODE integrator."""
        pass

    def step(self):
        if not self.is_inputs_ready():
            return
        self.ode_transition()

    @abstractmethod
    def ode_transition(self) -> Dict[str, Any]:
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
            """Initialize the CasADiIntegrator.

            Args:
                state: State to integrate.
                inputs: Input dependencies.
                step_size: Integration step size.
                state_dynamics_function: System dynamics function.
                dynamic_variable_paths: Paths to dynamic state variables.
            """
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


class TransistorModifier(ABC):
    """Base class for transistor modifiers that operate at the class level.

    Each modifier takes a Transistor class and returns a new subclass that adds or modifies behavior.
    By working at the class level, we ensure that `with_transistor()` can instantiate the final
    transistor class at runtime with the desired modifications.
    """

    @abstractmethod
    def apply_class(self, transistor_cls: Type["Transistor"]) -> Type["Transistor"]:
        """Given a transistor class, return a new transistor class with the modifier applied.

        Args:
            transistor_cls: The original transistor class to wrap.

        Returns:
            A new transistor class that includes the modifications.
        """
        pass


class SampleAndHoldModifier(TransistorModifier):
    """Class-level modifier that applies a sample-and-hold (zero-order-hold) behavior to the transistor.

    The resulting class updates the node's state at fixed intervals and holds the last computed state in between.
    """

    def __init__(self, hold_duration: float):
        """Initialize the SampleAndHoldModifier.

        Args:
            hold_duration: The duration to hold the state between updates.
        """
        self.hold_duration = hold_duration

    def apply_class(self, transistor_cls: Type["Transistor"]) -> Type["Transistor"]:
        hold_duration = self.hold_duration

        class SampleAndHoldTransistor(transistor_cls):
            """A transistor class modified with sample-and-hold behavior."""

            is_modifier = True

            def __init__(self, node, time_final: Optional[float] = None):
                super().__init__(node=node, time_final=time_final)
                self.last_update_time = None

            def step(self):
                # Retrieve clock if available
                clock_state = next(
                    (s for s in self.node.inputs.states if s.name == "Clock"), None
                )
                if clock_state is None or clock_state.value["value"] is None:
                    # No clock, just run the step each time
                    return super().step()

                current_time = float(clock_state.value["value"])

                if (
                    self.last_update_time is None
                    or current_time >= self.last_update_time + hold_duration
                ):
                    # Time to recompute the state
                    super().step()
                    self.last_update_time = current_time
                else:
                    # Hold previous state: do nothing
                    pass

        return SampleAndHoldTransistor


class ResetModifier(TransistorModifier):
    """Class-level modifier that adds reset logic.

    Checks a reset signal each step, and if triggered, resets the node state before proceeding.
    """

    def __init__(self, reset_path: Optional[str] = None):
        self.reset_path = reset_path

    def apply_class(self, transistor_cls: Type["Transistor"]) -> Type["Transistor"]:
        reset_path = self.reset_path

        class ResetTransistor(transistor_cls):
            """A transistor class modified with reset behavior."""

            is_modifier = True

            def __init__(self, node, time_final: Optional[float] = None):
                super().__init__(node=node, time_final=time_final)
                self.reset_signal_path = reset_path or f"reset_{self.node.state.name}"

            def step(self):
                if self.node.inputs[self.reset_signal_path].data:
                    self.node.reset()
                    return {}

                # Proceed with normal step
                return super().step()

        return ResetTransistor
