from regelum.environment.node.base import Node, State, Inputs
from regelum.environment.graph import Graph, visualize_graph
from regelum.environment.node.continuous import Pendulum
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from regelum.utils import rg
from stable_baselines3.common.buffers import ReplayBuffer
import gymnasium as gym
import torch.optim as optim
from typing import Optional
import os


class Observer(Node):
    state = State("observation", (3,), np.zeros(3))
    inputs = ["pendulum_state"]

    cos_max = sin_max = 1.0
    max_speed = np.inf
    observation_high = np.array([cos_max, sin_max, max_speed], dtype=np.float32)
    observation_space = gym.spaces.Box(
        low=-observation_high, high=observation_high, dtype=np.float32
    )

    def compute_state_dynamics(self) -> Dict[str, Any]:
        angle, velocity = self.inputs["pendulum_state"].data
        x = np.cos(angle)
        y = np.sin(angle)
        return {"observation": np.array([x, y, velocity])}


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


class RewardComputer(Node):
    def __init__(self, step_size: float = 0.1):
        state = State("reward", (1,), np.zeros(1))
        inputs = ["observation", "actor/action"]
        super().__init__(step_size=step_size, state=state, inputs=inputs)

    def compute_state_dynamics(self) -> Dict[str, Any]:
        state = self.inputs["observation"].data
        action = self.inputs["actor/action"].data

        cos_angle, sin_angle, angular_velocity = state
        angle = np.arctan2(sin_angle, cos_angle)
        position_cost = 1 * angle**2
        velocity_cost = 0.1 * angular_velocity**2
        control_cost = 0.01 * (action**2)
        reward = -(position_cost + velocity_cost + control_cost)

        return {"reward": np.array([reward])}


class Pendulum(Node):
    state = State(
        "pendulum_state",
        (2,),
        np.array([np.pi, 0]),
        _reset_modifier=lambda x: np.array(
            [
                np.random.uniform(-np.pi, np.pi),
                np.random.uniform(-1.0, 1.0),
            ]
        ),
    )
    inputs = Inputs(["actor/action", "reset_pendulum_state"])
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
        pendulum_mpc_control = self.inputs["actor/action"].data

        return self.system_dynamics(self.state.data, pendulum_mpc_control)


class ActorNetwork(nn.Module):
    def __init__(
        self,
        action_space: gym.Space,
        observation_space: gym.Space,
        log_std_min: float = -5,
        log_std_max: float = 2,
    ):
        super().__init__()
        self.fc1 = nn.Linear(np.array(observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(action_space.shape))
        # action rescalingd
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (action_space.high - action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (action_space.high + action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class Actor(Node):
    action_shape = (1,)
    max_torque = 2.0
    action_space = gym.spaces.Box(
        low=-max_torque, high=max_torque, shape=action_shape, dtype=np.float32
    )

    def __init__(
        self,
        step_size: float = 0.01,
        device="cpu",
        learning_starts: int = 5000,
    ):
        state = State(
            "actor",
            _value=[
                State("action", shape=self.action_shape),
                State(
                    "net",
                    _value=ActorNetwork(
                        action_space=self.action_space,
                        observation_space=Observer.observation_space,
                    ).to(device),
                ),
            ],
        )
        self.device = device
        inputs = Inputs(["observation", "step_counter"])
        self.learning_starts = learning_starts
        super().__init__(step_size=step_size, state=state, inputs=inputs)

    def compute_state_dynamics(self) -> Dict[str, Any]:
        if self.inputs["step_counter"].data < self.learning_starts:
            return {"actor/action": self.action_space.sample()}
        observation = (
            torch.FloatTensor(self.inputs["observation"].data)
            .reshape(1, -1)
            .to(self.device)
        )
        with torch.no_grad():
            action, _, _ = self.state["actor/net"].data.get_action(observation)
        return {"actor/action": action.cpu().numpy().reshape(-1)}


class Buffer(Node):
    def __init__(self, buffer_size: int = int(1e6), device: str = "cpu", **kwargs):

        self.buffer = ReplayBuffer(
            buffer_size=buffer_size,
            observation_space=Observer.observation_space,
            action_space=Actor.action_space,
            device=device,
            handle_timeout_termination=False,
        )
        state = State("replay_buffer", None, self.buffer)
        inputs = ["observation", "actor/action", "reward"]
        super().__init__(state=state, inputs=inputs, **kwargs)

        # Initialize replay buffer

        self.prev_obs = None

    def compute_state_dynamics(self) -> Dict[str, Any]:
        obs = self.inputs["observation"].data.reshape(1, -1)
        action = self.inputs["actor/action"].data.reshape(1, -1)
        reward = self.inputs["reward"].data.reshape(1, -1)
        replay_buffer: ReplayBuffer = self.state["replay_buffer"].data
        if self.prev_obs is not None:
            # Add transition to buffer
            replay_buffer.add(
                obs=self.prev_obs,
                next_obs=obs,
                action=action,
                reward=reward,
                done=np.array([False]),
                infos={},
            )

        self.prev_obs = obs
        return {}


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(observation_space.shape).prod()
            + np.array(action_space.shape).prod(),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AdaptationBlock(Node):

    def __init__(
        self,
        learning_starts: int = 5000,
        device: str = "cpu",
        autotune: bool = False,
        policy_lr: float = 0.0003,
        q_lr: float = 0.0003,
        alpha: Optional[float] = None,
        batch_size: int = 256,
        gamma: float = 0.99,
        policy_frequency: int = 2,
        target_network_frequency: int = 2,
        tau: float = 0.005,
    ):
        state = State("adaptation_block_summary", _value={})
        inputs = Inputs(["actor/net", "step_counter", "replay_buffer"])
        super().__init__(state=state, inputs=inputs)

        self.learning_starts = learning_starts
        self.device = device
        self.qf1 = SoftQNetwork(
            observation_space=Observer.observation_space,
            action_space=Actor.action_space,
        ).to(device)
        self.qf2 = SoftQNetwork(
            observation_space=Observer.observation_space,
            action_space=Actor.action_space,
        ).to(device)
        self.qf1_target = SoftQNetwork(
            observation_space=Observer.observation_space,
            action_space=Actor.action_space,
        ).to(device)
        self.qf2_target = SoftQNetwork(
            observation_space=Observer.observation_space,
            action_space=Actor.action_space,
        ).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=q_lr
        )

        self.actor_optimizer = None

        self.batch_size = batch_size
        self.gamma = gamma
        self.policy_lr = policy_lr
        self.policy_frequency = policy_frequency
        self.target_network_frequency = target_network_frequency
        self.autotune = autotune
        self.tau = tau
        # Automatic entropy tuning
        if autotune:
            self.target_entropy = -torch.prod(
                torch.Tensor(Actor.action_space.shape).to(self.device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=q_lr)
        else:
            assert alpha is not None
            self.alpha = alpha

    def compute_state_dynamics(self):
        if self.actor_optimizer is None:
            self.actor_optimizer = optim.Adam(
                list(self.inputs["actor/net"].data.parameters()), lr=self.policy_lr
            )
        if self.inputs["step_counter"].data < self.learning_starts:
            return {}

        data = self.inputs["replay_buffer"].data.sample(self.batch_size)
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.inputs[
                "actor/net"
            ].data.get_action(data.observations)
            qf1_next_target = self.qf1_target(
                data.next_observations, next_state_actions
            )
            qf2_next_target = self.qf2_target(
                data.next_observations, next_state_actions
            )
            min_qf_next_target = (
                torch.min(qf1_next_target, qf2_next_target)
                - self.alpha * next_state_log_pi
            )
            next_q_value = data.rewards.flatten() + (
                1 - data.dones.flatten()
            ) * self.gamma * (min_qf_next_target).view(-1)

        qf1_a_values = self.qf1(data.observations, data.actions).view(-1)
        qf2_a_values = self.qf2(data.observations, data.actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        # optimize the model
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        if self.inputs["step_counter"].data % self.policy_frequency == 0:
            for _ in range(self.policy_frequency):
                pi, log_pi, _ = self.inputs["actor/net"].data.get_action(
                    data.observations
                )
                qf1_pi = self.qf1(data.observations, pi)
                qf2_pi = self.qf2(data.observations, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                if self.autotune:
                    with torch.no_grad():
                        _, log_pi, _ = self.inputs["actor/net"].data.get_action(
                            data.observations
                        )
                    alpha_loss = (
                        -self.log_alpha.exp() * (log_pi + self.target_entropy)
                    ).mean()

                self.a_optimizer.zero_grad()
                alpha_loss.backward()
                self.a_optimizer.step()
                self.alpha = self.log_alpha.exp().item()

        if self.inputs["step_counter"].data % self.target_network_frequency == 0:
            for param, target_param in zip(
                self.qf1.parameters(), self.qf1_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            for param, target_param in zip(
                self.qf2.parameters(), self.qf2_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
        return {}


class PlotObservations(Node):
    def __init__(self, plot_dir: str = "plots", activate: bool = False):
        state = State("plot_counter", (1,), np.array([0]))
        inputs = ["observation", "is_truncated", "step_counter"]
        super().__init__(state=state, inputs=inputs)

        self.plot_dir = plot_dir
        os.makedirs(plot_dir, exist_ok=True)
        self.observations = []
        self.activate = activate

    def compute_state_dynamics(self) -> Dict[str, Any]:
        if not self.activate:
            return {}

        obs = self.inputs["observation"].data
        is_truncated = self.inputs["is_truncated"].data
        step = self.inputs["step_counter"].data[0]

        self.observations.append(obs)

        if is_truncated:
            import matplotlib.pyplot as plt

            observations = np.array(self.observations)
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

            # Plot cos(theta)
            ax1.plot(observations[:, 0], label="cos(θ)")
            ax1.set_ylabel("cos(θ)")
            ax1.grid(True)
            ax1.legend()

            # Plot sin(theta)
            ax2.plot(observations[:, 1], label="sin(θ)")
            ax2.set_ylabel("sin(θ)")
            ax2.grid(True)
            ax2.legend()

            # Plot angular velocity
            ax3.plot(observations[:, 2], label="ω")
            ax3.set_xlabel("Step")
            ax3.set_ylabel("Angular Velocity")
            ax3.grid(True)
            ax3.legend()

            plt.tight_layout()
            plt.savefig(f"{self.plot_dir}/episode_{step}.png")
            plt.close()

            # Reset observations for next episode
            self.observations = []

            return {"plot_counter": self.state.data + 1}

        return {}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
total_timesteps: int = 1000000
gamma: float = 0.99
"""the discount factor gamma"""
tau: float = 0.005
"""target smoothing coefficient (default: 0.005)"""
batch_size: int = 256
"""the batch size of sample from the reply memory"""
learning_starts: int = 5e3
"""timestep to start learning"""
policy_lr: float = 3e-4
"""the learning rate of the policy network optimizer"""
q_lr: float = 1e-3
"""the learning rate of the Q network network optimizer"""
policy_frequency: int = 2
"""the frequency of training policy (delayed)"""
target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
"""the frequency of updates for the target nerworks"""
alpha: float = 0.2
"""Entropy regularization coefficient."""
autotune: bool = True
"""automatic tuning of the entropy coefficient"""

pendulum = Pendulum(is_root=True, is_continuous=True)
observer = Observer()
actor = Actor(step_size=0.05, device=device, learning_starts=learning_starts)
is_truncated = IsTruncated(steps_to_truncate=200, step_size=0.03)
reset = Reset(input_node=pendulum)
reward_computer = RewardComputer()
buffer = Buffer(device=device)
adaptation_block = AdaptationBlock(
    gamma=gamma,
    tau=tau,
    batch_size=batch_size,
    learning_starts=learning_starts,
    policy_lr=policy_lr,
    q_lr=q_lr,
    policy_frequency=policy_frequency,
    target_network_frequency=target_network_frequency,
    alpha=alpha,
    autotune=autotune,
    device=device,
)
plot_observations = PlotObservations(activate=False)

graph = Graph(
    [
        pendulum,
        actor,
        adaptation_block,
        reward_computer,
        buffer,
        plot_observations,
        reset,
        is_truncated,
        observer,
    ]
)
subgraph = graph.extract_subgraph(
    "pendulum_state -> actor", freezed=[adaptation_block, reset, is_truncated]
)

subgraphs = subgraph.multiply(n_copies=2)

# data_buffers = []
# for sg in subgraphs:
#     selected = sg.select_nodes_contain(["pendulum_state", "actor/action"])
#     ith_databuffer = Buffer(selected)
#     sg.attach(ith_databuffer)
#     data_buffers.append(ith_databuffer)

graph.insert(subgraphs)

visualize_graph(graph, output_file="rl_pipeline_parallel", view=False)
