from regelum import Node, Graph
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from regelum.environment.node.nodes.classic_control.envs.continuous.pendulum import (
    Pendulum,
)
from regelum.environment.node.nodes.reset import Reset


class PendulumPDController(Node):

    def __init__(self, kp: float = 0.01, kd: float = 0.01, step_size: float = 0.01):
        super().__init__(
            inputs=["pendulum_1.pendulum_state"],
            step_size=step_size,
            is_root=True,
            name="pendulum_pd_controller",
        )
        self.kp = kp
        self.kd = kd

        self.pendulum_pd_control = self.define_variable(
            "pendulum_pd_control",
            value=np.array([0.0]),
            shape=(1,),
        )

    def step(self):
        pendulum_state = self.resolved_inputs.find("pendulum_1.pendulum_state").value

        angle = pendulum_state[0]
        angular_velocity = pendulum_state[1]

        self.pendulum_pd_control.value = -self.kp * angle - self.kd * angular_velocity


class PendulumReset(Reset):
    def __init__(self, reset_interval: int):
        super().__init__(name="reset_pendulum_1", inputs=["step_counter_1.counter"])
        self.reset_interval = reset_interval
        self.step_counter = 0

    def step(self) -> None:
        step_counter = self.resolved_inputs.find("step_counter_1.counter").value
        self.flag.value = step_counter % self.reset_interval == 0


class PlotDumper(Node):
    def __init__(self, save_dir: str = "plots", n_steps: int = 1000, **kwargs):
        inputs = [
            "pendulum_1.pendulum_state",
            "pendulum_pd_controller_1.pendulum_pd_control",
        ]
        super().__init__(inputs=inputs, is_root=True, **kwargs)
        self.n_steps = n_steps
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.states = []
        self.controls = []

    def step(self) -> None:
        self.states.append(
            self.resolved_inputs.find("pendulum_1.pendulum_state").value.copy()
        )
        self.controls.append(
            self.resolved_inputs.find(
                "pendulum_pd_controller_1.pendulum_pd_control"
            ).value
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

pd_controller = PendulumPDController(20, 20, step_size=0.01)
pendulum = Pendulum(control_signal_name="pendulum_pd_controller_1.pendulum_pd_control")
reset_pendulum = PendulumReset(reset_interval=1000)
plot_dumper = PlotDumper(n_steps=n_steps)
graph = Graph(
    [pd_controller, pendulum, reset_pendulum, plot_dumper],
    states_to_log=[
        "pendulum_1.pendulum_state",
        "pendulum_pd_controller_1.pendulum_pd_control",
    ],
    initialize_inner_time=True,
    logger_cooldown=0.2,
)
graph.resolve(graph.variables)


for _ in range(n_steps):
    graph.step()
