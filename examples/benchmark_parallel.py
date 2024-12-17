"""Benchmark parallel execution of node graphs."""

from regelum.environment.node.base_new import Node, Graph, Clock, Logger, StepCounter
import numpy as np
from numpy.typing import NDArray
import time
from statistics import mean, stdev
from typing import List, Dict, Any
import matplotlib.pyplot as plt


class Pendulum(Node):
    """Pendulum system dynamics."""

    def __init__(self, step_size: float = 0.001) -> None:
        """Initialize Pendulum node.

        Args:
            step_size: Integration time step
        """
        super().__init__(
            inputs=["controller.action"],
            step_size=step_size,
            is_continuous=True,
            is_root=True,
            name="pendulum",
        )

        # Physical parameters
        self.length = 1.0
        self.mass = 1.0
        self.gravity_acceleration = 9.81

        # State variable
        self.state = self.define_variable(
            "state",
            value=np.array([np.pi, 0.0]),
            metadata={
                "shape": (2,),
            },
        )

    def state_transition_map(self, state: NDArray, action: NDArray) -> NDArray:
        """Compute state derivatives."""
        angle, angular_velocity = state
        torque = action[0]  # Assuming 1D control input

        d_angle = angular_velocity
        d_angular_velocity = self.gravity_acceleration / self.length * np.sin(
            angle
        ) + torque / (self.mass * self.length**2)

        return np.array([d_angle, d_angular_velocity])

    def step(self) -> None:
        """Update pendulum state using Euler integration."""
        # Heavier computational load
        # result = 0.0
        # for _ in range(50000):  # Much more iterations
        #     result += np.sin(np.random.random()) * np.cos(np.random.random())

        action = self.resolved_inputs.find("controller.action").value
        derivatives = self.state_transition_map(self.state.value, action)
        self.state.value += self.step_size * derivatives


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
            inputs=["pendulum.state"],
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
        # Add computational load
        # result = 0.0
        # for _ in range(50000):  # Match pendulum load
        #     result += np.sin(np.random.random()) * np.cos(np.random.random())

        pendulum_state = self.resolved_inputs.find("pendulum.state").value
        angle = pendulum_state[0]
        angular_velocity = pendulum_state[1]
        self.action.value[0] = -self.kp * angle - self.kd * angular_velocity


def create_pendulum_graph(
    num_pairs: int = 5, debug: bool = False
) -> tuple[Graph, list[Node]]:
    """Create a graph with multiple pendulum-controller pairs.

    Args:
        num_pairs: Number of pendulum-controller pairs

    Returns:
        Tuple of (graph, list of nodes)
    """
    # Create base nodes
    clock = Clock(0.01)
    counter = StepCounter([clock], start_count=0)

    # Create pendulum-controller pairs
    pendulums = []
    controllers = []
    log_vars = ["clock_1.time", "step_counter_1.counter"]

    for i in range(num_pairs):
        pendulum = Pendulum(step_size=0.01)
        pendulum.variables[0].set_new_value(np.array([np.pi + np.random.random(), 0.0]))
        controller = PendulumPDController(kp=20.0, kd=10.1, step_size=0.01)

        pendulum.alter_input_names({"controller.action": f"controller_{i+1}.action"})
        controller.alter_input_names({"pendulum.state": f"pendulum_{i+1}.state"})

        pendulums.append(pendulum)
        controllers.append(controller)
        log_vars.extend(
            [f"{pendulum.external_name}.state", f"{controller.external_name}.action"]
        )

    # Create logger
    logger = Logger(variables_to_log=log_vars, step_size=0.01)

    # Build graph
    nodes = [clock, counter, logger]
    nodes.extend(pendulums)
    nodes.extend(controllers)

    return Graph(nodes, debug=debug), nodes


def run_simulation(graph: Graph, n_steps: int = 10) -> Dict[str, List[Any]]:
    """Run simulation and collect trajectory data."""
    trajectories: Dict[str, List[Any]] = {}

    # Initialize trajectories dict
    logger_node = next(n for n in graph.nodes if isinstance(n, Logger))
    for var_name in logger_node.variables_to_log:
        trajectories[var_name] = []

    # Run simulation and collect data
    for _ in range(n_steps):
        graph.step()
        # Get data directly from resolved inputs
        for var_name in logger_node.variables_to_log:
            var = logger_node.resolved_inputs.find(var_name)
            if var is not None:
                trajectories[var_name].append(np.array(var.value))

    return trajectories


def plot_trajectories(
    sequential_traj: Dict[str, List[Any]],
    parallel_traj: Dict[str, List[Any]],
    num_pairs: int,
) -> None:
    """Plot pendulum trajectories from both sequential and parallel simulations.

    Args:
        sequential_traj: Trajectories from sequential simulation
        parallel_traj: Trajectories from parallel simulation
        num_pairs: Number of pendulum-controller pairs
    """
    time_seq = sequential_traj["clock_1.time"]
    time_par = parallel_traj["clock_1.time"]

    fig, axes = plt.subplots(num_pairs, 2, figsize=(15, 4 * num_pairs))
    if num_pairs == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_pairs):
        # Plot angles
        ax_angle = axes[i, 0]
        seq_angle = [state[0] for state in sequential_traj[f"pendulum_{i+1}.state"]]
        par_angle = [state[0] for state in parallel_traj[f"pendulum_{i+1}.state"]]

        ax_angle.plot(time_seq, seq_angle, "b-", label="Sequential")
        ax_angle.plot(time_par, par_angle, "r--", label="Parallel")
        ax_angle.set_title(f"Pendulum {i+1} Angle")
        ax_angle.set_xlabel("Time [s]")
        ax_angle.set_ylabel("Angle [rad]")
        ax_angle.legend()
        ax_angle.grid(True)

        # Plot actions
        ax_action = axes[i, 1]
        seq_action = sequential_traj[f"controller_{i+1}.action"]
        par_action = parallel_traj[f"controller_{i+1}.action"]

        ax_action.plot(time_seq, seq_action, "b-", label="Sequential")
        ax_action.plot(time_par, par_action, "r--", label="Parallel")
        ax_action.set_title(f"Controller {i+1} Action")
        ax_action.set_xlabel("Time [s]")
        ax_action.set_ylabel("Torque")
        ax_action.legend()
        ax_action.grid(True)

    plt.tight_layout()
    plt.savefig("trajectories.pdf")
    plt.close()


def benchmark_execution(
    n_iterations: int = 5, debug: bool = False, num_pairs: int = 3, n_steps: int = 10
) -> tuple[List[float], List[float]]:
    """Benchmark parallel vs non-parallel execution.

    Args:
        n_iterations: Number of iterations for timing
        debug: Whether to enable debug mode
        num_pairs: Number of pendulum-controller pairs
        n_steps: Number of simulation steps

    Returns:
        Tuple of (sequential_times, parallel_times)
    """
    sequential_times = []
    parallel_times = []

    # Store trajectories from last iteration
    last_sequential_traj = None
    last_parallel_traj = None

    # Create and resolve graph once
    graph, nodes = create_pendulum_graph(num_pairs=num_pairs, debug=debug)
    graph.resolve(graph.variables)

    for iteration in range(n_iterations):
        print(f"\nIteration {iteration + 1}/{n_iterations}")

        # Reset all nodes
        for node in nodes:
            node.reset()

        # Time sequential execution
        print("Running sequential simulation...")
        start_time = time.perf_counter()
        sequential_traj = run_simulation(graph, n_steps)
        seq_time = time.perf_counter() - start_time
        sequential_times.append(seq_time)
        print(f"Sequential execution time: {seq_time:.6f} seconds")

        if iteration == n_iterations - 1:
            last_sequential_traj = sequential_traj

        graph.reset()

        # Create parallel version and time it
        print("Running parallel simulation...")
        parallel_graph = graph.parallelize()

        start_time = time.perf_counter()
        parallel_traj = run_simulation(parallel_graph, n_steps)
        par_time = time.perf_counter() - start_time
        parallel_times.append(par_time)
        print(f"Parallel execution time: {par_time:.6f} seconds")

        if iteration == n_iterations - 1:
            last_parallel_traj = parallel_traj

        # Clean up parallel resources
        parallel_graph.close()
        parallel_graph.reset()

    # Plot trajectories from the last iteration
    if last_sequential_traj and last_parallel_traj:
        print("\nPlotting trajectories from the last iteration...")
        plot_trajectories(last_sequential_traj, last_parallel_traj, num_pairs)
        print("Trajectories saved to 'trajectories.pdf'")

    return sequential_times, parallel_times


def print_benchmark_results(
    sequential_times: List[float], parallel_times: List[float]
) -> None:
    """Print benchmark results with statistics.

    Args:
        sequential_times: List of sequential execution times
        parallel_times: List of parallel execution times
    """
    print("\nBenchmark Results")
    print("=" * 80)

    # Calculate statistics
    seq_mean = mean(sequential_times)
    seq_std = stdev(sequential_times) if len(sequential_times) > 1 else 0
    par_mean = mean(parallel_times)
    par_std = stdev(parallel_times) if len(parallel_times) > 1 else 0
    speedup = seq_mean / par_mean if par_mean > 0 else float("inf")

    # Print results
    print(f"\nSequential Execution:")
    print(f"  Mean time: {seq_mean:.6f} seconds")
    print(f"  Std dev:   {seq_std:.6f} seconds")
    print(f"\nParallel Execution:")
    print(f"  Mean time: {par_mean:.6f} seconds")
    print(f"  Std dev:   {par_std:.6f} seconds")
    print(f"\nSpeedup: {speedup:.2f}x")

    print("\nDetailed timings:")
    print("\nIteration  Sequential    Parallel    Speedup")
    print("-" * 45)
    for i, (seq_t, par_t) in enumerate(zip(sequential_times, parallel_times), 1):
        iter_speedup = seq_t / par_t if par_t > 0 else float("inf")
        print(f"{i:^9d}  {seq_t:^10.6f}  {par_t:^10.6f}  {iter_speedup:^7.2f}x")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    NUM_PAIRS = 10
    N_STEPS = 500  # Increased number of steps for better visualization

    print("\nRunning benchmarks...")
    sequential_times, parallel_times = benchmark_execution(
        n_iterations=1, debug=True, num_pairs=NUM_PAIRS, n_steps=N_STEPS
    )
    print_benchmark_results(sequential_times, parallel_times)
