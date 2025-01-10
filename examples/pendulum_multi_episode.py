from regelum.node.base import Node
from regelum.node.graph import Graph
from regelum.node.classic_control.envs.continuous import Pendulum
from regelum.node.classic_control.controllers.pid import PIDControllerBase
from regelum.node.memory.buffer import DataBuffer
import numpy as np
import matplotlib
from typing import List

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class PlotDumper(Node):
    """Node for collecting and plotting trajectories."""

    def __init__(
        self, inputs: List[str], n_trajectories: int = 100, step_size: float = 0.01
    ) -> None:

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

        for input_name in self.inputs.inputs:
            buffer_data = self.resolved_inputs.find(input_name).value
            plt.plot(buffer_data[:, 0], buffer_data[:, 1], alpha=0.3)

        plt.xlabel("Angle")
        plt.ylabel("Angular Velocity")
        plt.title(f"Phase Portrait of {self.n_trajectories} Pendulum Trajectories")
        plt.grid(True)
        plt.savefig("./examples/gfx/pendulum_trajectories.png")
        plt.close()


def main():

    pendulum = Pendulum(
        control_signal_name="pid_controller_1.control_signal",
        state_reset_modifier=lambda x: x + np.random.randn(2) * 0.1,
    )
    controller = PIDControllerBase(
        controlled_state=pendulum.state,
        idx_controlled_state=0,
        kp=10,
        ki=0.0,
        kd=10,
        step_size=0.01,
    )
    data_buffer = DataBuffer(
        variable_full_names=[
            pendulum.state.full_name,
            controller.control_signal.full_name,
        ],
        buffer_sizes=[300, 300],
        step_sizes=[0.01, 0.01],
    )

    nodes = [pendulum, controller, data_buffer]
    graph = Graph(
        nodes,
        debug=True,
        initialize_inner_time=True,
        states_to_log=[pendulum.state.full_name],
        logger_cooldown=0.3,
    )
    graph.resolve(graph.variables)
    subgraph = graph.extract_as_subgraph(
        [
            "step_counter_1",
            "clock_1",
            "pendulum_1",
            "pid_controller_1",
            "data_buffer_1",
            "logger_1",
        ],
        n_step_repeats=300,
    )

    for node in subgraph.nodes:
        for var in node.variables:
            var.reset(apply_reset_modifier=True)

    # Batch clone all nodes at once
    cloned_graphs = []
    for i in range(199):
        cloned = graph.clone_node(f"graph_2", defer_resolution=True)
        cloned_graphs.append(cloned)
        for node in cloned.nodes:
            for var in node.variables:
                var.reset(apply_reset_modifier=True)

    # Resolve all at once after cloning
    graph.resolve_all_clones()

    plot_dumper = PlotDumper(
        inputs=[
            var
            for var in graph.get_full_names()
            if "pendulum" in var and "buffer" in var
        ],
        n_trajectories=200,
        step_size=0.01,
    )
    graph.insert_node(plot_dumper)
    parallel_graph = graph.parallelize()
    parallel_graph.step()


if __name__ == "__main__":
    main()
