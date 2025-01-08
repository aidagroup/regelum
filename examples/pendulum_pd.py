import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from regelum import Node, Graph
from regelum.node.classic_control.envs.continuous import Pendulum
from regelum.node.classic_control.controllers.pid import (
    PIDControllerBase,
)
from regelum.node.reset import ResetEachNSteps


class PlotDumper(Node):
    def __init__(self, save_dir: str = "plots", n_steps: int = 1000, **kwargs):
        inputs = [
            "pendulum_1.state",
            "pid_controller_1.control_signal",
        ]
        super().__init__(inputs=inputs, is_root=True, **kwargs)
        self.n_steps = n_steps
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.states = []
        self.controls = []

    def step(self) -> None:
        self.states.append(self.resolved_inputs.find("pendulum_1.state").value.copy())
        self.controls.append(
            self.resolved_inputs.find("pid_controller_1.control_signal").value
        )

        if len(self.states) == self.n_steps:
            self._dump_plot()

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


pendulum = Pendulum(control_signal_name="pid_controller_1.control_signal")
pd_controller = PIDControllerBase(pendulum.state, 0, kp=20, ki=0, kd=20, step_size=0.01)
reset_pendulum = ResetEachNSteps(node_name_to_reset=pendulum.external_name, n_steps=100)
plot_dumper = PlotDumper(n_steps=n_steps)
graph = Graph(
    [pendulum, pd_controller, reset_pendulum, plot_dumper],
    states_to_log=[
        "pendulum_1.state",
        "pid_controller_1.control_signal",
    ],
    initialize_inner_time=True,
    logger_cooldown=0.2,
)
graph.resolve(graph.variables)


for _ in range(n_steps):
    graph.step()
