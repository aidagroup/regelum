"""Module that contains general objectives functions that can be used by various entities of the framework.

For instance, a running objective can be used commonly by a generic optimal scenario, an actor, a critic, a logger, an animator, a scenario etc.

"""

from abc import ABC, abstractmethod

import regelum
from .model import Model, PerceptronWithTruncatedNormalNoise, ModelNN
from typing import Optional
import torch
from .utils import rg
from .predictor import Predictor
from regelum.typing import RgArray


class Objective(regelum.RegelumBase, ABC):
    """A base class for objective implementations."""

    def __init__(self):
        """Initialize an instance of Objective."""
        pass

    @abstractmethod
    def __call__(self):
        pass


class RunningObjective(Objective):
    """Running reward/cost.

    In minimzations problems, it is called cost or loss, say.
    """

    def __init__(self, model: Optional[Model] = None):
        """Initialize a RunningObjective instance.

        Args:
            model (function): function that calculates the running
                objective for a given observation and action.
        """
        self.model = (lambda observation, action: 0) if model is None else model

    def __call__(
        self, observation: RgArray, action: RgArray, is_save_batch_format: bool = False
    ) -> RgArray:
        """
        Calculate the running objective for a given observation and action, potentially formatting the output.

        This method computes the objective value based on the current state of the environment as represented
        by the observation and the action taken by an agent. It can also preprocess the output to be in batch
        format if specified.

        Args:
            observation: The current observation from the environment, which should reflect the
                current state. This is typically a numpy array or an array-like structure that contains
                numerical data representing the state of the environment the agent is interacting with.
            action: The action taken by the agent in response to the observation. Similar to the
                observation, this is typically a numpy array or an array-like structure containing numerical
                data that represents the action chosen by the agent.
            is_save_batch_format: A flag to determine if the output should be preprocessed
                into a batch format. When False, the method does not alter the output. When True, it will
                preprocess the output into a format that is suitable for batch processing, which is often
                required for machine learning models that process data in batches. Default is False.

        Returns:
            The computed running objective value, which is a numerical score representing the
                performance or the reward of the agent given the observation and the action taken.
                This value is usually used in reinforcement learning to guide the agent's learning process.
        """

        running_objective = self.model(observation, action)
        if not is_save_batch_format:
            return running_objective[0][0]

        return running_objective


class RunningObjectivePendulum(RunningObjective):
    def __call__(
        self, observation: RgArray, action: RgArray, is_save_batch_format: bool = False
    ) -> RgArray:
        model_res = super().__call__(observation, action, is_save_batch_format)
        return (
            model_res + 15 * observation[0, 0] ** 2
            if abs(observation[0, 0]) > 0.3
            else model_res + 15 * observation[0, 1] ** 2
        )


def reinforce_objective(
    policy_model: PerceptronWithTruncatedNormalNoise,
    observations: torch.FloatTensor,
    actions: torch.FloatTensor,
    tail_values: torch.FloatTensor,
    values: torch.FloatTensor,
    baselines: torch.FloatTensor,
    is_with_baseline: bool,
    is_do_not_let_the_past_distract_you: bool,
    N_episodes: int,
) -> torch.FloatTensor:
    """Calculate the surrogate objective for REINFORCE algorithm.

    Args:
        policy_model: The policy model that outputs log probabilities.
        observations: Observed states from the environment.
        actions: Actions taken in response to the observations.
        tail_values: The rewards-to-go or costs-to-go for each step in the episode.
        values: The total accumulated running objectives for each step in the episode.
        baselines: The baseline values used for variance reduction.
        is_with_baseline: If True, subtract baselines from the target objectives.
        is_do_not_let_the_past_distract_you: If True, use tail values instead of total values.
        N_episodes: The count of episodes over which the average is taken.

    Returns:
        The computed surrogate objective value as a float tensor.
    """
    log_pdfs = policy_model.log_pdf(observations, actions)
    if is_do_not_let_the_past_distract_you:
        target_objectives = tail_values
    else:
        target_objectives = values
    if is_with_baseline:
        target_objectives -= baselines

    return (log_pdfs * target_objectives).sum() / N_episodes


def get_gae_advantage(
    gae_lambda: float,
    running_objectives: torch.FloatTensor,
    values: torch.FloatTensor,
    times: torch.FloatTensor,
    discount_factor: float,
    sampling_time: float,
) -> torch.FloatTensor:
    """Calculate the Generalized Advantage Estimation (GAE) advantage.

    Args:
        gae_lambda: The GAE lambda parameter.
        running_objectives: The running objectives tensor.
        values: The critic values tensor.
        times: The timestamps tensor.
        discount_factor: The discount factor.
        sampling_time: The sampling time between actions

    Returns:
        GAE advantage tensor.
    """
    deltas = (
        running_objectives[:-1]
        + discount_factor**sampling_time * values[1:]
        - values[:-1]
    )
    if gae_lambda == 0.0:
        advantages = deltas
    else:
        gae_discount_factors = (gae_lambda * discount_factor) ** times[:-1]
        reversed_gae_discounted_deltas = torch.flip(
            gae_discount_factors * deltas, dims=[0, 1]
        )
        advantages = (
            torch.flip(reversed_gae_discounted_deltas.cumsum(dim=0), dims=[0, 1])
            / gae_discount_factors
        )
    return advantages


def sdpg_objective(
    policy_model: PerceptronWithTruncatedNormalNoise,
    critic_model: ModelNN,
    observations: torch.FloatTensor,
    actions: torch.FloatTensor,
    times: torch.FloatTensor,
    episode_ids: torch.LongTensor,
    discount_factor: float,
    N_episodes: int,
    running_objectives: torch.FloatTensor,
    sampling_time: float,
    is_normalize_advantages: bool,
    gae_lambda: float,
) -> torch.FloatTensor:
    """Calculate the surrogate objective for the Stochastic Deterministic Policy Gradient (SDPG) algorithm.

    Args:
        policy_model: The policy model that outputs log probabilities.
        critic_model: The critic model that estimates Q-values for observation-action pairs.
        observations: A tensor of observed states from the environment.
        actions: A tensor of actions taken by the agent.
        times: A tensor of time steps corresponding to each observation-action pair.
        episode_ids: A tensor of episode identifiers.
        discount_factor: A scalar discount factor for future rewards, typically between 0 and 1.
        N_episodes: The number of episodes to average over for the objective calculation.
        running_objectives: A tensor of running objective values up to the current time step.
        sampling_time: The time step used for sampling in the environment.
        is_normalize_advantages: If True, normalize the advantage estimates.
        gae_lambda: The lambda parameter for GAE, controlling the trade-off between bias and variance.


    Returns:
        SDPG surrogate objective.
    """
    critic_values = critic_model(observations)
    log_pdfs = policy_model.log_pdf(observations, actions).reshape(-1)

    objective = 0.0
    for episode_idx in torch.unique(episode_ids):
        mask = (episode_ids == episode_idx).reshape(-1)
        advantages = get_gae_advantage(
            gae_lambda=gae_lambda,
            running_objectives=running_objectives[mask],
            values=critic_values[mask],
            times=times[mask],
            discount_factor=discount_factor,
            sampling_time=sampling_time,
        ).reshape(-1)

        if is_normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        objective += (
            discount_factor ** times[mask][:-1].reshape(-1)
            * advantages
            * log_pdfs[mask][:-1]
        ).sum()

    return objective / N_episodes


def ppo_objective(
    policy_model: PerceptronWithTruncatedNormalNoise,
    critic_model: ModelNN,
    observations: torch.FloatTensor,
    actions: torch.FloatTensor,
    times: torch.FloatTensor,
    episode_ids: torch.LongTensor,
    discount_factor: float,
    N_episodes: int,
    running_objectives: torch.FloatTensor,
    cliprange: float,
    initial_log_probs: torch.FloatTensor,
    running_objective_type: str,
    sampling_time: float,
    gae_lambda: float,
    is_normalize_advantages: bool = True,
    entropy_coeff: float = 0.0,
) -> torch.FloatTensor:
    """Calculate PPO objective.

    Args:
        policy_model: The neural network model representing the policy.
        critic_model: The neural network model representing the value function (critic).
        observations: A tensor of observations from the environment.
        actions: A tensor of actions taken by the agent.
        times: A tensor with timestamps for each observation-action pair.
        episode_ids: A tensor with unique identifiers for each episode.
        discount_factor: The factor by which future rewards are discounted.
        N_episodes: The total number of episodes over which the objective is averaged.
        running_objectives: A tensor of accumulated rewards or costs for each timestep.
        cliprange: The range for clipping the probability ratio in the objective function.
        initial_log_probs: The log probabilities of taking the actions at the time
            of sampling, under the policy model before the update.
        running_objective_type (str): Indicates whether the running objectives are 'cost' or 'reward'.
        sampling_time: The timestep used for sampling in the environment.
        gae_lambda: The lambda parameter for GAE, controlling the trade-off between bias and variance.
        is_normalize_advantages: Flag indicating whether to normalize advantage estimates.

    Returns:
        objective for PPO
    """
    assert (
        running_objective_type == "cost" or running_objective_type == "reward"
    ), "running_objective_type can be either 'cost' or 'reward'"

    critic_values = critic_model(observations)
    prob_ratios = torch.exp(
        policy_model.log_pdf(observations, actions) - initial_log_probs.reshape(-1)
    ).reshape(-1, 1)
    if hasattr(policy_model, "entropy"):
        entropies = entropy_coeff * policy_model.entropy(observations).reshape(-1, 1)
    else:
        entropies = torch.zeros_like(prob_ratios)
    clipped_prob_ratios = torch.clamp(prob_ratios, 1 - cliprange, 1 + cliprange)
    objective_value = 0.0
    for episode_idx in torch.unique(episode_ids):
        mask = episode_ids.reshape(-1) == episode_idx
        advantages = get_gae_advantage(
            gae_lambda=gae_lambda,
            running_objectives=running_objectives[mask],
            values=critic_values[mask],
            times=times[mask],
            discount_factor=discount_factor,
            sampling_time=sampling_time,
        )
        if is_normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        objective_value += (
            torch.sum(
                (discount_factor ** times[mask][:-1])
                * (
                    torch.maximum(
                        advantages * prob_ratios[mask][:-1],
                        advantages * clipped_prob_ratios[mask][:-1],
                    )
                    - entropies[mask][:-1]
                    if running_objective_type == "cost"
                    else torch.minimum(
                        advantages * prob_ratios[mask][:-1],
                        advantages * clipped_prob_ratios[mask][:-1],
                    )
                    + entropies[mask][:-1]
                )
            )
            / N_episodes
        )

    return objective_value


def ddpg_objective(
    policy_model: ModelNN,
    critic_model: ModelNN,
    observations: torch.FloatTensor,
) -> torch.FloatTensor:
    """Calculate the objective value for the DDPG algorithm.

    Args:
        policy_model: The policy model that generates actions based on observations.
        critic_model: The critic model that approximates the Q-function.
        observations: The batch of observations.

    Returns:
        torch.FloatTensor: The objective value.
    """
    return critic_model(
        observations, policy_model.forward(observations, is_means_only=True)
    ).mean()


def temporal_difference_objective_full(
    critic_model_output: ModelNN,
    running_objective: torch.FloatTensor,
    td_n: int,
    discount_factor: float,
    sampling_time: float,
    episode_ids: torch.LongTensor,
    critic_targets: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    objective = 0.0
    n_iterations = torch.unique(episode_ids).shape[0]
    for episode_id in torch.unique(episode_ids):
        mask = (episode_ids == episode_id).reshape(-1)
        objective += (
            temporal_difference_objective(
                critic_model_output=critic_model_output[mask],
                running_objective=running_objective[mask],
                td_n=td_n,
                discount_factor=discount_factor,
                sampling_time=sampling_time,
                critic_targets=(
                    critic_targets[mask] if critic_targets is not None else None
                ),
            )
            / n_iterations
        )
    return objective


def temporal_difference_objective(
    critic_model_output: ModelNN,
    running_objective: torch.FloatTensor,
    td_n: int,
    discount_factor: float,
    sampling_time: float,
    critic_targets: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """Calculate temporal difference objective.

    Args:
        critic_model (ModelNN): Q function model.
        observation (torch.FloatTensor): Batch of observations.
        action (torch.FloatTensor): Batch of actions.
        running_objective (torch.FloatTensor): Batch of running
            objectives.
        td_n (int): Number of temporal difference steps.
        discount_factor (float): Discount factor for future running
            objectives.
        device (Union[str, torch.device]): Device to proceed
            computations on.
        sampling_time (float): Sampling time for discounting.
        is_use_same_critic (bool): Whether to use the critic model from
            the previous iteration or not (`True` for not using, `False`
            for using).

    Returns:
        torch.FloatTensor: objective value
    """
    batch_size = running_objective.shape[0]
    assert batch_size > td_n, f"batch size {batch_size} too small for such td_n {td_n}"
    discount_factors = rg.array(
        [[discount_factor ** (sampling_time * i)] for i in range(td_n)],
        prototype=running_objective,
        _force_numeric=True,
    )
    discounted_tdn_sum_of_running_objectives = rg.vstack(
        [
            rg.sum(running_objective[i : i + td_n, :] * discount_factors)
            for i in range(batch_size - td_n)
        ]
    )

    if critic_targets is None:
        critic_targets = critic_model_output[td_n:, :]
    else:
        critic_targets = critic_targets[td_n:, :]

    temporal_difference = rg.mean(
        (
            critic_model_output[:-td_n, :]
            - discounted_tdn_sum_of_running_objectives
            - discount_factor ** (td_n * sampling_time) * critic_targets
        )
        ** 2
    )
    return temporal_difference


def mpc_objective(
    observation,
    estimated_state,
    policy_model_weights,
    predictor: Predictor,
    running_objective,
    model,
    prediction_horizon,
    discount_factor=1.0,
):
    (
        state_sequence_predicted,
        action_sequence_predicted,
    ) = predictor.predict_state_sequence_from_model(
        estimated_state,
        prediction_horizon=prediction_horizon,
        model=model,
        model_weights=policy_model_weights,
        is_predict_last=False,
    )
    observation_sequence_predicted = predictor.system.get_observation(
        None,
        state=state_sequence_predicted,
        inputs=action_sequence_predicted,
        is_batch=True,
    )
    observation_sequence = rg.vstack(
        (rg.force_row(observation), observation_sequence_predicted)
    )

    running_objectives = running_objective(
        observation_sequence, action_sequence_predicted, is_save_batch_format=True
    )

    if discount_factor < 1.0:
        discount_factors = rg.array(
            [
                [discount_factor ** (predictor.pred_step_size * i)]
                for i in range(prediction_horizon + 1)
            ],
            prototype=observation,
            _force_numeric=True,
        )
        mpc_objective_value = rg.sum(running_objectives * discount_factors)
    else:
        mpc_objective_value = rg.sum(running_objectives)

    return mpc_objective_value


def rpv_objective(
    observation,
    estimated_state,
    policy_model_weights,
    predictor: Predictor,
    running_objective,
    model,
    prediction_horizon,
    discount_factor,
    critic,
    critic_weights,
):
    rpv_objective_value = 0
    (
        state_sequence_predicted,
        action_sequence_predicted,
    ) = predictor.predict_state_sequence_from_model(
        estimated_state,
        prediction_horizon=prediction_horizon,
        model=model,
        model_weights=policy_model_weights,
        is_predict_last=True,
    )
    observation_sequence_predicted = predictor.system.get_observation(
        None,
        state=state_sequence_predicted,
        inputs=action_sequence_predicted,
        is_batch=True,
    )
    observation_sequence = rg.vstack(
        (rg.force_row(observation), observation_sequence_predicted[:-1, :])
    )

    discount_factors = rg.array(
        [
            [discount_factor ** (predictor.pred_step_size * i)]
            for i in range(prediction_horizon)
        ],
        prototype=observation,
        _force_numeric=True,
    )

    rpv_objective_value = rg.sum(
        discount_factors
        * running_objective(
            observation_sequence, action_sequence_predicted, is_save_batch_format=True
        )
    )

    observation_last = observation_sequence_predicted[-1, :]
    rpv_objective_value += rg.sum(
        discount_factor ** (predictor.pred_step_size * prediction_horizon)
        * critic(observation_last, weights=critic_weights)
    )

    return rpv_objective_value


def rql_objective(
    observation,
    estimated_state,
    policy_model_weights,
    predictor: Predictor,
    running_objective,
    model,
    prediction_horizon,
    discount_factor,
    critic,
    critic_weights,
):
    rql_objective_value = 0
    (
        state_sequence_predicted,
        action_sequence_predicted,
    ) = predictor.predict_state_sequence_from_model(
        estimated_state,
        prediction_horizon=prediction_horizon,
        model=model,
        model_weights=policy_model_weights,
        is_predict_last=False,
    )

    if prediction_horizon == 0:
        return critic(observation, action_sequence_predicted, weights=critic_weights)
    observation_sequence_predicted = predictor.system.get_observation(
        None,
        state=state_sequence_predicted,
        inputs=action_sequence_predicted,
        is_batch=True,
    )
    observation_sequence = rg.vstack(
        (rg.force_row(observation), observation_sequence_predicted[:-1, :])
    )
    discount_factors = rg.array(
        [
            [discount_factor ** (predictor.pred_step_size * i)]
            for i in range(prediction_horizon)
        ],
        prototype=observation,
        _force_numeric=True,
    )
    rql_objective_value = rg.sum(
        discount_factors
        * running_objective(
            observation_sequence,
            action_sequence_predicted[:-1, :],
            is_save_batch_format=True,
        )
    )

    observation_last = state_sequence_predicted[-1, :]
    rql_objective_value += rg.sum(
        discount_factor ** (predictor.pred_step_size * prediction_horizon)
        * critic(
            observation_last, action_sequence_predicted[-1, :], weights=critic_weights
        )
    )

    return rql_objective_value


def sql_objective(
    observation,
    estimated_state,
    policy_model_weights,
    predictor: Predictor,
    model,
    prediction_horizon,
    critic,
    critic_weights=None,
):
    (
        state_sequence_predicted,
        action_sequence_predicted,
    ) = predictor.predict_state_sequence_from_model(
        estimated_state,
        prediction_horizon=prediction_horizon,
        model=model,
        model_weights=policy_model_weights,
        is_predict_last=False,
    )
    observation_sequence_predicted = predictor.system.get_observation(
        None,
        state=state_sequence_predicted,
        inputs=action_sequence_predicted,
        is_batch=True,
    )
    observation_sequence = rg.vstack(
        (rg.force_row(observation), observation_sequence_predicted)
    )

    sql_objective_value = rg.sum(
        rg.sum(
            critic(
                observation_sequence, action_sequence_predicted, weights=critic_weights
            )
        )
    )

    return sql_objective_value
