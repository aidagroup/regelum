from regelum.node.base import Node
from regelum.node.graph import Graph
from regelum.node.classic_control.envs.continuous import Pendulum
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class PendulumPDController(Node):
    """PD controller for pendulum."""

    def __init__(
        self, kp: float = 0.01, kd: float = 0.01, step_size: float = 0.01
    ) -> None:
        """Initialize PD controller.

        Args:
            kp: Proportional gain
            kd: Derivative gain
            step_size: Control update interval
        """
        super().__init__(
            inputs=["pendulum_1.state"],
            step_size=step_size,
            is_continuous=False,
            name="controller",
        )

        self.kp = kp
        self.kd = kd

        # Control output
        self.action = self.define_variable(
            "action",
            value=np.array([0.0]),
            metadata={"shape": (1,)},
        )

    def step(self) -> None:
        """Compute control action using PD law."""
        pendulum_state = self.resolved_inputs.find("pendulum.state").value
        angle = pendulum_state[0]
        angular_velocity = pendulum_state[1]
        self.action.value[0] = -self.kp * angle - self.kd * angular_velocity


class DataBuffer(Node):
    """Buffer for storing trajectory data."""

    def __init__(self, buffer_size: int = 1000, step_size: float = 0.01) -> None:
        super().__init__(
            inputs=["pendulum_1.state", "step_counter_1.counter"],
            step_size=step_size,
            is_continuous=False,
            name="buffer",
        )
        self.buffer_size = buffer_size

        self.state_buffer = self.define_variable(
            "state_buffer",
            value=np.zeros((buffer_size, 2)),
            metadata={"shape": (buffer_size, 2)},
        )

    def step(self) -> None:
        """Store current state in buffer."""
        current_buffer_idx = self.resolved_inputs.find("step_counter.counter").value - 1
        current_state = self.resolved_inputs.find("pendulum.state").value
        self.state_buffer.value[current_buffer_idx] = current_state


class PlotDumper(Node):
    """Node for collecting and plotting trajectories."""

    def __init__(self, n_trajectories: int = 100, step_size: float = 0.01) -> None:
        inputs = [f"buffer_{i+1}.state_buffer" for i in range(n_trajectories)]
        super().__init__(
            inputs=inputs,
            step_size=step_size,
            is_continuous=False,
            name="plot_dumper",
        )
        self.n_trajectories = n_trajectories
        self.plot_data = self.define_variable(
            "plot_data",
            value=False,
            metadata={"shape": (1,)},
        )

    def step(self) -> None:
        """Collect data and create plot."""
        plt.figure(figsize=(10, 6))

        for i in range(self.n_trajectories):
            buffer_data = self.resolved_inputs.find(f"buffer_{i+1}.state_buffer").value
            plt.plot(buffer_data[:, 0], buffer_data[:, 1], alpha=0.3)

        plt.xlabel("Angle")
        plt.ylabel("Angular Velocity")
        plt.title(f"Phase Portrait of {self.n_trajectories} Pendulum Trajectories")
        plt.grid(True)
        plt.savefig("pendulum_trajectories.png")
        plt.close()


def main():

    controller = PendulumPDController(kp=10, kd=10, step_size=0.01)
    pendulum = Pendulum(
        control_signal_name=controller.action.full_name,
        state_reset_modifier=lambda x: x + np.random.randn(2) * 0.1,
    )
    data_buffer = DataBuffer(step_size=0.01, buffer_size=300)

    # Create initial graph
    nodes = [pendulum, controller, data_buffer]
    graph = Graph(
        nodes,
        debug=True,
        initialize_inner_time=True,
        states_to_log=["pendulum_1.state"],
        logger_cooldown=0.1,
    )
    graph.resolve(graph.variables)
    subgraph = graph.extract_as_subgraph(
        [
            "step_counter_1",
            "clock_1",
            "pendulum_1",
            "controller_1",
            "buffer_1",
            "logger_1",
        ],
        n_step_repeats=300,
    )

    for node in subgraph.nodes:
        for var in node.variables:
            var.reset(apply_reset_modifier=True)

    for _ in range(4):
        cloned = graph.clone_node("graph_2")
        for node in cloned.nodes:
            for var in node.variables:
                var.reset(apply_reset_modifier=True)

    plot_dumper = PlotDumper(n_trajectories=5, step_size=0.01)
    graph.insert_node(plot_dumper)
    parallel_graph = graph.parallelize()
    parallel_graph.step()
    graph.print_info()


if __name__ == "__main__":
    main()
