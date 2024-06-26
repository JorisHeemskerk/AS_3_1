from memory import Memory
from policy import Policy
from transition import Transition
from state import State
from doubleAgent import DoubleAgent
from action import Action

import torch
from torch import nn
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random


class DoubleAgent2(DoubleAgent):
    """
    double deep q-learning agent
    """

    def __init__(
        self, policy: Policy, 
        memory: Memory, 
        target_policy: Policy,
        tau: float
    ) -> None:
        super().__init__(policy, memory, target_policy=target_policy, tau=tau)
        self.epsilon = 1

    def decay(self):
        self.epsilon *= 0.996

    def select_action(self, state: State)-> Action:
        """
        Select action based on current policy.
        
        @param state: Current State to perform action in.

        @return Action with Action to perform.
        """
        if random.random() < self.epsilon:
            return random.choice(list(Action))
        return Action(torch.argmax(self.policy._network(torch.Tensor(tuple(state)))).item())

    def train(
        self,
        gamma: float, 
        memory_batch_size: int,
    )-> None:
        if self.memory._size <= memory_batch_size:
            return
        
        batch: list[Transition] = self.memory.get_batch(
            batch_size=memory_batch_size
        )

        X = []
        Y = []

        for state, action, reward, state_prime, terminal in batch:
            # print(type(state))
            # print(type(state_prime))
            # print(type(tuple(state_prime)))
            q_prime = list(self.target_policy._network(torch.Tensor(tuple(state_prime))))
            a_prime = q_prime.index(max(q_prime))
            a_value = reward + (gamma * q_prime[a_prime]) * (1 - terminal)

            q_state = self.policy._network(torch.Tensor(tuple(state)))
            X.append(q_state)

            q_state = list(q_state)
            q_state[action.value] = a_value

            Y.append(q_state)

        # Compute loss
        loss = self.policy.loss(torch.stack(X), torch.Tensor(Y))

        # Zero the gradients
        self.policy.optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Optimizer step
        self.policy.optimizer.step()

        # Update target network params
        self.align_target_network()