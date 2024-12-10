from regelum.environment.node.base import Node, State, Inputs
from regelum.environment.graph import Graph
import numpy as np
from regelum.utils import rg
import logging
from typing import Dict, Any
from pathlib import Path
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class Pendulum(Node):
    state = State("pendulum_state", (2,), np.array([np.pi, 0]))
    inputs = Inputs(["pendulum_pd_control", "reset_pendulum_state"])
    length = 1
    mass = 1
    gravity_acceleration = 9.81

    def system_dynamics(self, x, u):
        pendulum_mpc_control = u

        angle = x[0]
        angular_velocity = x[1]
        torque = pendulum_mpc_control

        d_angle = angular_velocity
        d_angular_velocity = self.gravity_acceleration / (self.length) * rg.sin(
            angle
        ) + torque / (self.mass * self.length**2)

        return {"pendulum_state": rg.vstack([d_angle, d_angular_velocity])}

    def compute_state_dynamics(self):
        pendulum_mpc_control = self.inputs["pendulum_pd_control"].data

        return self.system_dynamics(self.state.data, pendulum_mpc_control)


class PendulumPDController(Node):
    state = State("pendulum_pd_control", (1,))
    inputs = Inputs(["pendulum_state"])

    def __init__(self, kp: float = 0.01, kd: float = 0.01, step_size: float = 0.01):
        super().__init__(step_size=step_size)
        self.kp = kp
        self.kd = kd

    def compute_state_dynamics(self):
        pendulum_state = self.inputs["pendulum_state"].data

        angle = pendulum_state[0]
        angular_velocity = pendulum_state[1]

        return {"pendulum_pd_control": -self.kp * angle - self.kd * angular_velocity}


class IsTruncated(Node):
    def __init__(self, steps_to_truncate: int, **kwargs):
        state = State("is_truncated", (1,), False)
        inputs = ["step_counter"]
        self.steps_to_truncate = steps_to_truncate
        super().__init__(state=state, inputs=inputs, **kwargs)

    def compute_state_dynamics(self) -> Dict[str, Any]:
        residual = self.inputs["step_counter"].data[0] % self.steps_to_truncate
        return {self.state.name: residual == 0}


class Reset(Node):
    def __init__(self, input_node: Node, **kwargs):
        state = State(f"reset_{input_node.state.name}", (1,), False)
        inputs = ["is_truncated"]
        super().__init__(state=state, inputs=inputs, **kwargs)

    def compute_state_dynamics(self) -> Dict[str, Any]:
        return {self.state.name: self.inputs["is_truncated"].data}


class PlotDumper(Node):
    def __init__(self, save_dir: str = "plots", **kwargs):
        state = State("plot_dumper", (1,), False)
        inputs = ["pendulum_state", "pendulum_pd_control"]
        super().__init__(state=state, inputs=inputs, **kwargs)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.states = []
        self.controls = []

    def compute_state_dynamics(self) -> Dict[str, Any]:
        self.states.append(self.inputs["pendulum_state"].data)
        self.controls.append(self.inputs["pendulum_pd_control"].data)

        if len(self.states) == n_steps:  # Using global n_steps
            self._dump_plot()

        return {self.state.name: False}

    def _dump_plot(self) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        states = np.array(self.states)
        controls = np.array(self.controls)
        times = np.arange(len(states)) * self.step_size

        ax1.plot(times, states[:, 0], label="angle")
        ax1.plot(times, states[:, 1], label="angular velocity")
        ax1.set_ylabel("State")
        ax1.legend()

        ax2.plot(times, controls, label="control")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Control")
        ax2.legend()

        plt.savefig(self.save_dir / "simulation.png")
        plt.close()


n_steps = 1000

pd_controller = PendulumPDController(20, 20, step_size=0.01)
pendulum = Pendulum(is_root=True, is_continuous=True)
is_truncated = IsTruncated(steps_to_truncate=1000)
reset = Reset(pendulum)
plot_dumper = PlotDumper()
graph = Graph(
    [pd_controller, pendulum, is_truncated, reset, plot_dumper],
    states_to_log=["pendulum_state", "pendulum_pd_control"],
)


for _ in range(n_steps):
    graph.step()
