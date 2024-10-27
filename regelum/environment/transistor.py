"""This module contains the base class for all transistors that compute transitions of nodes composing an environment."""

from __future__ import annotations
from typing import Callable, Optional, Dict, Any
from abc import abstractmethod, ABC
from regelum.environment.node import Node, State, Inputs
from casadi import integrator, MX, vertcat, vec, DM
import numpy as np


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
        step_size: float = 1.0,  # default step size
        time_final: Optional[float] = None,
    ) -> None:
        """Instantiate a Transistor with the associated Node."""
        self.node = node
        self.step_size = step_size
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
        step_size: float,
        time_final: Optional[float] = None,
        time_start: float = 0.0,
    ) -> None:
        """Instantiate an ODETransistor."""
        super().__init__(node=node, step_size=step_size, time_final=time_final)
        self.time = self.time_start = time_start
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

        def create(self):

            import regelum as rg

            with rg.symbolic_inference():
                inputs_symbolic_dict = self.inputs_info.collect()
                inputs_symbolic_vector = (
                    vertcat(*[vec(k) for k in inputs_symbolic_dict.values()])
                    if inputs_symbolic_dict
                    else MX([])
                )
                state_dynamics = self.state_dynamics_function()

                DAE = {
                    "x": self.state_info.value["value"],
                    "p": inputs_symbolic_vector,
                    "ode": list(state_dynamics.values())[0],
                }

            # options = {"tf": self.step_size}
            return integrator("integrator", "rk", DAE, 0, self.step_size)

    def setup_integrator(self):
        self.integrator_interface = self.CasADiIntegrator(
            state=self.node.state,
            inputs=self.node.inputs,
            step_size=self.step_size,
            state_dynamics_function=self.node.compute_state_dynamics,
        )
        self.integrator = self.integrator_interface.create()

    def ode_transition(self) -> Dict[str, Any]:
        """Compute the new state using the CasADi integrator."""
        inputs = self.collect_inputs()
        # Prepare the initial state
        x0 = self.node.state.value["value"]
        if isinstance(x0, np.ndarray):
            x0 = DM(x0)
        else:
            x0 = DM([x0])

        inputs_values = [
            inputs[state.name]["value"] for state in self.node.inputs.states
        ]
        p = vertcat(*[vec(DM(val)) for val in inputs_values])

        res = self.integrator(x0=x0, p=p)
        xf = res["xf"]
        new_state_value = np.array(xf.full()).flatten()
        return {self.node.state.name: new_state_value}
