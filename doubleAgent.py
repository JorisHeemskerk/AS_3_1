from memory import Memory
from policy import Policy
from transition import Transition
from state import State
from agent import Agent

import torch
from torch import nn
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


class DoubleAgent(Agent):
    """
    double deep q-learning agent
    """

    def __init__(
        self, policy: Policy, 
        memory: Memory, 
        target_policy: Policy,
        tau: float
    ) -> None:
        super().__init__(policy, memory)

        self.target_policy = target_policy
        self.tau = tau

    def train_batch(
        self, 
        gamma: float, 
        memory_batch_size: int,
        batch_size: int,
        loss_fn: nn.Module=nn.CrossEntropyLoss(),
        optimizer: torch.optim.Optimizer=torch.optim.Adam
    )-> None:
        if self.memory._size <= memory_batch_size:
            return
        
        batch: list[Transition] = self.memory.get_batch(
            batch_size=memory_batch_size
        )

        train_Xs = []
        train_ys = []

        for transition in batch:
            next_state_q_values = self.target_policy.forward(
                transition.next_state
            )
        
            best_value = max(
                next_state_q_values
            ) if transition.terminated else 0

            current_state_q_values = self.policy.forward(
                transition.state
            )

            current_state_q_values[
                transition.action.value
            ] = transition.reward if transition.terminated else \
            transition.reward + gamma * best_value
            
            train_Xs.append(transition.state)
            train_ys.append(current_state_q_values)

        self.policy.train(
            X_train=train_Xs,
            y_train=train_ys,
            batch_size=batch_size,
            loss_fn=loss_fn,
            optimizer=optimizer
        )
        
        self.align_target_network()

    def align_target_network(self):
        for param, target_param in zip(
            self.policy._network.parameters(), 
            self.target_policy._network.parameters()
        ):
            target_param.data.copy_(
                self.tau * 
                param.data + 
                target_param.data * 
                (1.0 - self.tau)
            )

    def train(
        self, 
        environment: gym.Env,
        n_episodes: int,
        gamma: float, 
        memory_batch_size: int,
        batch_size: int,
        steps_limit:int,
        loss_fn: nn.Module=nn.CrossEntropyLoss(),
        optimizer: torch.optim.Optimizer=torch.optim.Adam,
        seed: int=42,
        )-> None:
        """
        Train the policy.

        @param environment: the environment
        @param n_iterations: number of iterations
        @param gamma: discount value
        @param memory_batch_size: number of samples from memory
        @param batch_size: batch size for training the model
        @param loss_fn: loss function, default=nn.CrossEntropyLoss
        @param optimizer: optimizer function, default=torch.optim.Adam
        @param n_episodes: number of episodes
        @param seed: random seed for the environment initialisation
        """
        self.rewards = []
        pbar = tqdm(range(n_episodes))
        for i in pbar:
            if i > 100 and np.average(self.rewards[-100:]) >= 200:
                print("Done training, it good enough d=====(￣▽￣*)b")
                return
            total_reward = 0
            start_state, _ = environment.reset(seed=seed)
            start_state = State(*start_state)
            step_number = 0
            while step_number < steps_limit:
                action = self.policy.select_action(start_state)
                state_prime, reward, is_terminated, truncated, _ = \
                    environment.step(action.value)
                state_prime = State(*state_prime)

                total_reward += reward
                
                self.memory.store(
                    Transition(
                        start_state, 
                        action,
                        reward,
                        state_prime,
                        is_terminated
                    )
                )

                if is_terminated or truncated:
                    break

                start_state = state_prime
                step_number+= 1           
                self.train_batch(
                    gamma=gamma,
                    memory_batch_size=memory_batch_size,
                    batch_size=batch_size,
                    loss_fn=loss_fn,
                    optimizer=optimizer
                ) 
            self.policy.decay_epsilon()
            self.target_policy.decay_epsilon()
            self.rewards.append(total_reward)
            
            pbar.set_postfix({'current reward': self.rewards[-1], "current epsilon": self.policy.epsilon})

