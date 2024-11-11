from regelum.environment.node.base import Node, State, Graph, Inputs
from regelum.environment.node.memory.data_buffer import DataBuffer
from regelum.environment.node.memory import create_memory_chain
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import List, Optional, Tuple
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
from regelum.utils import rg
import mlflow
import os


class PolicyNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        std: float = 0.5,
        action_bounds: Optional[List[List[float]]] = None,
        noise_std: float = 0.05,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim),  # Add LayerNorm
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim),  # Add LayerNorm
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.ones(1) * np.log(std))
        self.action_bounds = action_bounds
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> Normal:
        mean = self.net(x)
        std = self.log_std.exp()
        dist = Normal(mean, std)
        return dist

    def sample_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.forward(state)
        action_pre_tanh = dist.rsample()
        noise = torch.randn_like(action_pre_tanh) * self.noise_std
        action_pre_tanh += noise
        action = torch.tanh(action_pre_tanh)

        # Correct log_prob calculation
        log_prob = dist.log_prob(action_pre_tanh)
        # Adjustment for tanh transformation
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # Sum over action dimensions

        if self.action_bounds is not None:
            bounds = torch.tensor(self.action_bounds)
            action = (
                action * (bounds[:, 1] - bounds[:, 0]) / 2
                + (bounds[:, 1] + bounds[:, 0]) / 2
            )

        return action, log_prob


class ValueNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim),  # Add LayerNorm
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(hidden_dim),  # Add LayerNorm
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class IsDone(Node):
    def __init__(
        self,
        max_episode_steps: int = 200,
        max_angle: float = 4 * np.pi,  # ~720 degrees
        max_velocity: float = 10.0,
        step_size: float = 0.1,
    ):
        self.max_episode_steps = max_episode_steps
        self.max_angle = max_angle
        self.max_velocity = max_velocity
        state = State(
            "is_done",
            None,
            [
                State("flag", (1,), np.array([False])),
                State("episode_steps", (1,), np.array([0])),
            ],
        )
        inputs = ["pendulum_state"]  # Add pendulum state input
        super().__init__(
            step_size=step_size,
            state=state,
            inputs=inputs,
        )

    def compute_state_dynamics(self) -> Dict[str, Any]:
        steps = self.state["is_done/episode_steps"].data[0]
        angle, velocity = self.inputs["pendulum_state"].data

        # Check all termination conditions
        is_done = (
            steps >= self.max_episode_steps
            or abs(angle) > self.max_angle
            or abs(velocity) > self.max_velocity
        )

        steps = 0 if is_done else steps + 1
        return {
            "is_done/flag": np.array([is_done]),
            "is_done/episode_steps": np.array([steps]),
        }


class ReinforceController(Node):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        step_size: float = 0.1,
        noise_std: float = 0.05,
    ):
        self.policy = PolicyNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            std=0.5,
            action_bounds=[[-17.0, 17.0]],
            noise_std=noise_std,
        )

        state = State(
            "reinforce_controller",
            None,
            [
                State("control", (1,), np.zeros(1)),
                State("action_pre_tanh", (1,), np.zeros(1)),
            ],
        )

        inputs = ["summator"]

        super().__init__(
            step_size=step_size,
            state=state,
            inputs=inputs,
        )

    def compute_state_dynamics(self) -> Dict[str, Any]:
        observation = self.inputs["summator"].data
        policy = self.policy

        with torch.no_grad():
            state_tensor = torch.FloatTensor(observation)
            action, _ = policy.sample_action(state_tensor)
            action_pre_tanh = torch.atanh(torch.clamp(action, -0.999999, 0.999999))
            action = action.numpy()
            action_pre_tanh = action_pre_tanh.numpy()

        return {
            "reinforce_controller/control": action,
            "reinforce_controller/action_pre_tanh": action_pre_tanh,
        }


class ReinforceUpdater(Node):
    def __init__(
        self,
        controller: ReinforceController,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
        batch_size: int = 5,
        step_size: float = 0.1,
        input_dim: int = 3,
    ):
        self.controller = controller
        self.policy = controller.policy
        state = State(
            "reinforce_updater",
            None,
            [
                State("episode_return", (1,), np.zeros(1)),
                State("episode_count", (1,), np.zeros(1)),
            ],
        )
        inputs = [
            "buffer/summator_buffer",
            "buffer/action_pre_tanh_buffer",
            "buffer/reward_buffer",
            "is_done/flag",
            "step_counter",
        ]

        self.optimizer = Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.returns = []
        self.value_losses = []
        self.batch_size = batch_size
        self.episodes_collected = 0
        self.all_states = []
        self.all_actions_pre_tanh = []
        self.all_rewards = []
        self.value_network = ValueNetwork(input_dim=input_dim, hidden_dim=40)
        self.value_optimizer = Adam(self.value_network.parameters(), lr=learning_rate)

        super().__init__(
            step_size=step_size,
            state=state,
            inputs=inputs,
        )

    def reset(self):
        # Override reset to prevent resetting optimizer and networks
        self.state.reset()
        self.episodes_collected = 0
        self.all_states = []
        self.all_actions_pre_tanh = []
        self.all_rewards = []

    def _compute_returns(self, rewards: np.ndarray) -> np.ndarray:
        returns = np.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        return returns

    def compute_state_dynamics(self) -> Dict[str, Any]:
        is_done = self.inputs["is_done/flag"].data[0]
        if is_done:
            # Collect data from the current episode
            rewards = self.inputs["buffer/reward_buffer"].data.copy()
            states = self.inputs["buffer/summator_buffer"].data.copy()
            actions_pre_tanh = self.inputs["buffer/action_pre_tanh_buffer"].data.copy()

            self.all_states.append(states)
            self.all_actions_pre_tanh.append(actions_pre_tanh)
            self.all_rewards.append(rewards)
            self.episodes_collected += 1

            episode_return = rewards.sum()
            self.returns.append(episode_return)
            self.state["reinforce_updater/episode_return"].data = np.array(
                [episode_return]
            )

            if self.episodes_collected >= self.batch_size:
                # Concatenate data from all collected episodes
                all_states = np.concatenate(self.all_states, axis=0)
                all_actions = np.concatenate(self.all_actions_pre_tanh, axis=0)
                all_rewards = np.concatenate(self.all_rewards, axis=0)

                # Update policy
                self._update_policy_with_data(all_states, all_actions, all_rewards)

                # Reset collections
                self.all_states = []
                self.all_actions_pre_tanh = []
                self.all_rewards = []
                self.episodes_collected = 0

            # Update episode count
            self.state["reinforce_updater/episode_count"].data += 1

        return {
            "reinforce_updater/episode_return": self.state[
                "reinforce_updater/episode_return"
            ].data,
            "reinforce_updater/episode_count": self.state[
                "reinforce_updater/episode_count"
            ].data,
        }

    def _update_policy_with_data(
        self, states: np.ndarray, actions_pre_tanh: np.ndarray, rewards: np.ndarray
    ):
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions_pre_tanh)
        returns_tensor = torch.FloatTensor(self._compute_returns(rewards))

        # Update value network
        values = self.value_network(states_tensor)
        value_loss = nn.MSELoss()(values, returns_tensor)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.value_losses.append(value_loss.item())

        # Compute advantages
        advantages = returns_tensor - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update policy network
        dist = self.policy(states_tensor)
        log_probs = dist.log_prob(actions_tensor)
        log_probs -= torch.log(1 - torch.tanh(actions_tensor).pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1)

        policy_loss = -(log_probs * advantages).mean()
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()


class Pendulum(Node):
    state = State(
        "pendulum_state",
        (2,),
        np.array([np.pi, 0]),
        _reset_modifier=lambda x: x + np.random.uniform(-0.5, 0.5, size=2),
    )
    inputs = Inputs(["reinforce_controller/control"])
    length = 1
    mass = 1
    gravity_acceleration = 9.81

    def system_dynamics(self, x, u):
        pendulum_mpc_control = u

        angle = x[0]
        angular_velocity = x[1]
        torque = pendulum_mpc_control

        d_angle = angular_velocity
        d_angular_velocity = self.gravity_acceleration / self.length * rg.sin(
            angle
        ) + torque / (self.mass * self.length**2)

        return {"pendulum_state": rg.vstack([d_angle, d_angular_velocity])}

    def compute_state_dynamics(self):
        pendulum_mpc_control = self.inputs["reinforce_controller/control"].data

        return self.system_dynamics(self.state.data, pendulum_mpc_control)


class EpisodeCounter(Node):
    def __init__(self, step_size: float = 0.1):
        state = State("episode_counter", (1,), np.zeros(1))
        inputs = ["is_done"]
        super().__init__(step_size=step_size, state=state, inputs=inputs)

    def compute_state_dynamics(self) -> Dict[str, Any]:
        is_done = self.inputs["is_done"].data[0]
        count = self.state.data[0]
        if is_done:
            count += 1
        return {"episode_counter": np.array([count])}


class UpdateCounter(Node):
    def __init__(self, update_freq: int = 1, step_size: float = 0.1):
        state = State("update_counter", (1,), np.zeros(1))
        inputs = ["episode_counter"]
        self.update_freq = update_freq
        super().__init__(step_size=step_size, state=state, inputs=inputs)

    def compute_state_dynamics(self) -> Dict[str, Any]:
        episode_count = self.inputs["episode_counter"].data[0]
        count = self.state.data[0]
        if episode_count > 0 and episode_count % self.update_freq == 0:
            count = episode_count / self.update_freq
        return {"update_counter": np.array([count])}


class RewardComputer(Node):
    def __init__(self, step_size: float = 0.1):
        state = State("reward", (1,), np.zeros(1))
        inputs = ["observer", "reinforce_controller/control"]
        super().__init__(step_size=step_size, state=state, inputs=inputs)

    def compute_state_dynamics(self) -> Dict[str, Any]:
        state = self.inputs["observer"].data
        action = self.inputs["reinforce_controller/control"].data

        x, y, velocity = state
        position_cost = 3 * (x**2 + y**2)  # Cost relative to upright position (0, -1)
        velocity_cost = 0.1 * velocity**2
        control_cost = 0.1 * (action**2)
        reward = -(position_cost + velocity_cost + control_cost)

        return {"reward": np.array([reward])}


class RewardPlotter(Node):
    def __init__(self, step_size: float = 0.1, window_size: int = 100):
        state = State(
            "reward_plotter",
            None,
            [
                State("episode_rewards", (window_size,), np.zeros(window_size)),
                State("current_idx", (1,), np.zeros(1)),
            ],
        )
        inputs = ["reinforce_updater/episode_return", "is_done/flag"]
        self.window_size = window_size
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        (self.line,) = self.ax.plot([], [])
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Episode Return")
        self.ax.set_title("Online Training Progress")
        self.ax.grid(True)
        plt.ion()
        super().__init__(step_size=step_size, state=state, inputs=inputs)

    def compute_state_dynamics(self) -> Dict[str, Any]:
        is_done = self.inputs["is_done/flag"].data[0]
        current_idx = int(self.state["reward_plotter/current_idx"].data[0])
        rewards = self.state["reward_plotter/episode_rewards"].data

        if is_done:
            episode_return = self.inputs["reinforce_updater/episode_return"].data[0]
            rewards[current_idx % self.window_size] = episode_return
            current_idx += 1

            if current_idx > 2:  # Start plotting after a few episodes
                self.line.set_data(range(len(rewards)), rewards)
                self.ax.relim()
                self.ax.autoscale_view()
                plt.pause(0.01)

        return {
            "reward_plotter/episode_rewards": rewards,
            "reward_plotter/current_idx": np.array([current_idx]),
        }


class MLflowLogger(Node):
    def __init__(
        self, step_size: float = 0.1, experiment_name: str = "pendulum_reinforce"
    ):
        state = State(
            "mlflow_logger",
            None,
            [
                State(
                    "episode_data", None, {"states": [], "actions": [], "returns": []}
                ),
                State("episode_count", (1,), np.zeros(1)),
            ],
        )
        inputs = [
            "pendulum_state",
            "reinforce_controller/control",
            "reinforce_updater/episode_return",
            "is_done/flag",
        ]

        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run()
        super().__init__(step_size=step_size, state=state, inputs=inputs)

    def reset(self):
        # Clear the episode data completely
        self.state["mlflow_logger/episode_data"].data = {
            "states": [],
            "actions": [],
            "returns": [],
        }

    def compute_state_dynamics(self) -> Dict[str, Any]:
        is_done = self.inputs["is_done/flag"].data[0]
        state = self.inputs["pendulum_state"].data
        action = self.inputs["reinforce_controller/control"].data
        episode_data = self.state["mlflow_logger/episode_data"].data

        # Collect current step data
        episode_data["states"].append(state.copy())
        episode_data["actions"].append(action.copy())

        if is_done:
            episode_count = self.state["mlflow_logger/episode_count"].data[0]
            episode_return = self.inputs["reinforce_updater/episode_return"].data[0]

            # Log metrics
            mlflow.log_metric("episode_return", episode_return, step=int(episode_count))

            # Create and save trajectory plot
            if episode_count % 10 == 0:  # Log every 10 episodes
                # Convert lists to numpy arrays for current episode only
                states = np.array(episode_data["states"])
                actions = np.array(episode_data["actions"])
                time = np.arange(len(states))

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

                # Plot states
                ax1.plot(time, states[:, 0], label="angle")
                ax1.plot(time, states[:, 1], label="angular velocity")
                ax1.set_xlabel("Step")
                ax1.set_ylabel("State")
                ax1.legend()
                ax1.grid(True)

                # Plot actions
                ax2.plot(time, actions, label="control")
                ax2.set_xlabel("Step")
                ax2.set_ylabel("Action")
                ax2.legend()
                ax2.grid(True)

                plt.tight_layout()

                # Save plot
                plot_path = f"episode_{int(episode_count)}_trajectory.png"
                plt.savefig(plot_path)
                plt.close()

                # Log artifact
                mlflow.log_artifact(plot_path)
                os.remove(plot_path)

                # Log state/action data as CSV
                trajectory_data = np.concatenate([states, actions], axis=1)
                np.savetxt(
                    f"episode_{int(episode_count)}_data.csv",
                    trajectory_data,
                    delimiter=",",
                    header="angle,angular_velocity,action",
                    comments="",
                )
                mlflow.log_artifact(f"episode_{int(episode_count)}_data.csv")
                os.remove(f"episode_{int(episode_count)}_data.csv")

            episode_count += 1
            self.state["mlflow_logger/episode_count"].data = np.array([episode_count])
            # Reset data after processing
            self.reset()

        return {
            "mlflow_logger/episode_data": episode_data,
            "mlflow_logger/episode_count": self.state[
                "mlflow_logger/episode_count"
            ].data,
        }

    def __del__(self):
        mlflow.end_run()


class Observer(Node):
    def __init__(self, step_size: float = 0.1):
        state = State("observer", (3,), np.zeros(3))  # [x, y, velocity]
        inputs = ["pendulum_state"]
        super().__init__(step_size=step_size, state=state, inputs=inputs)

    def compute_state_dynamics(self) -> Dict[str, Any]:
        angle, velocity = self.inputs["pendulum_state"].data
        x = np.sin(angle)
        y = 1 - np.cos(angle)
        return {"observer": np.array([x, y, velocity])}


class Summator(Node):
    def __init__(
        self,
        input_paths: List[str],
        dim: int,
        output_name: str = "summator",
        step_size: float = 0.1,
    ):
        state = State(output_name, (dim,), np.zeros(dim))
        super().__init__(
            step_size=step_size,
            state=state,
            inputs=input_paths,
        )

    def compute_state_dynamics(self) -> Dict[str, Any]:
        # Concatenate all inputs in order
        concatenated = np.concatenate(
            [np.atleast_1d(state.data).flatten() for state in self.inputs.states]
        )

        return {self.state.name: concatenated}


# Graph Creation
max_episode_steps = 300
step_size = 0.03
n_memory_cells = 3
n_episodes_in_batch = 5

pendulum = Pendulum(is_root=True, is_continuous=True)
is_done = IsDone(max_episode_steps=max_episode_steps)
controller = ReinforceController(
    step_size=step_size, noise_std=0.8, input_dim=(n_memory_cells + 1) * 3
)
reward_computer = RewardComputer(step_size=step_size)
updater = ReinforceUpdater(
    controller=controller,
    gamma=0.99,
    batch_size=n_episodes_in_batch,
    step_size=step_size,
    input_dim=(n_memory_cells + 1) * 3,
    learning_rate=0.0001,
)
reward_plotter = RewardPlotter(step_size=step_size, window_size=100)
observer = Observer(step_size=step_size)

mlflow_logger = MLflowLogger(step_size=step_size)

memory_chain = create_memory_chain(
    observer,
    n_memory_cells,
    ["observer"],
    step_size=step_size,
)

summator = Summator(
    input_paths=[
        "observer",
        "1_memory/previous/observer",
        "2_memory/previous/observer",
        "3_memory/previous/observer",
    ],
    dim=12,
    step_size=step_size,
)

data_buffer = DataBuffer(
    nodes_to_buffer=[pendulum, controller, reward_computer, summator],
    paths_to_remember=[
        "pendulum_state",
        "reinforce_controller/action_pre_tanh",
        "summator",
        "reward",
    ],
    buffer_size=max_episode_steps
    * n_episodes_in_batch,  # Adjust buffer size for one episode
)
graph = Graph(
    [
        pendulum,
        observer,
        controller,
        reward_computer,
        data_buffer,
        is_done,
        updater,
        reward_plotter,
        mlflow_logger,
        summator,
        *memory_chain,
    ],
    states_to_log=[
        "pendulum_state",
        "reinforce_controller/action_pre_tanh",
        "reward",
        "step_counter",
    ],
    logger_cooldown=10,
)


for _ in range(1000000):
    graph.step()
    if is_done.state["is_done/flag"].data[0]:
        # Now reset for the next episode
        pendulum.reset()
        is_done.reset()  # Resets 'is_done/flag' and 'episode_steps'
        data_buffer.reset()
        graph.reset(["Clock", "Logger/last_log_time", "Logger/counter"])
        mlflow_logger.reset()


print(
    f"Training completed after {updater.state['reinforce_updater/episode_count'].data[0]} episodes"
)

# Plotting Results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(updater.returns)
plt.xlabel("Episode")
plt.ylabel("Episode Return")
plt.title("REINFORCE Learning Curve")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(updater.value_losses)
plt.xlabel("Episode")
plt.ylabel("Value Loss")
plt.title("Value Network Learning Curve")
plt.grid(True)

plt.tight_layout()
plt.show()
