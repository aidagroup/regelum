"""SARSA (State-Action-Reward-State-Action) agent implementation.

This module implements the SARSA algorithm, an on-policy temporal difference learning
method for reinforcement learning. The agent learns by estimating Q-values for
state-action pairs through direct interaction with the environment, updating its
estimates based on the actual actions taken rather than the maximum Q-value of the
next state (as in Q-learning).

The agent follows an epsilon-greedy policy for action selection, balancing
exploration and exploitation during learning.
"""
