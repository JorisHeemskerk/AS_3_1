from memory import Memory
from policy import Policy
from transition import Transition
from state import State

import torch
from torch import nn
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


class Agent:
    """
    Agent class.
    """

    def __init__(self, policy: Policy, memory: Memory)-> None:
        """
        Initializer for Agent.

        @param policy: Agent's movement policy
        @param memory: Agent's memory
        """
        self.policy = policy
        self.memory = memory
        self.rewards = []

    def train_batch(
        self, 
        gamma: float, 
        memory_batch_size: int,
        batch_size: int,
        loss_fn: nn.Module=nn.CrossEntropyLoss(),
        )-> None:
        """
        Train the policy on a batch of transitions.

        @param gamma: discount value
        @param memory_batch_size: number of samples from memory
        @param batch_size: batch size for training the model
        @param loss_fn: loss function, default=nn.CrossEntropyLoss
        """
        # TODO: miss wel bewaren zo idk
        if self.memory._size <= memory_batch_size:
            return

        batch: list[Transition] = self.memory.get_batch(
            batch_size=memory_batch_size
        )

        train_Xs = []
        train_ys = []

        for transition in batch:
            next_state_q_values = self.policy.forward(
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
        )

    # def train(
    #     self, 
    #     environment: gym.Env,
    #     n_iterations: int,
    #     gamma: float, 
    #     memory_batch_size: int,
    #     batch_size: int,
    #     steps_limit:int,
    #     loss_fn: nn.Module=nn.CrossEntropyLoss(),
    #     optimizer: torch.optim.Optimizer=torch.optim.Adam,
    #     n_episodes: int=64,
    #     seed: int=42,
    #     )-> None:
    #     """
    #     Train the policy.

    #     @param environment: the environment
    #     @param n_iterations: number of iterations
    #     @param gamma: discount value
    #     @param memory_batch_size: number of samples from memory
    #     @param batch_size: batch size for training the model
    #     @param loss_fn: loss function, default=nn.CrossEntropyLoss
    #     @param optimizer: optimizer function, default=torch.optim.Adam
    #     @param n_episodes: number of episodes
    #     @param seed: random seed for the environment initialisation
    #     """
    #     self.rewards = []
    #     for i in tqdm(range(n_iterations), desc="iteration "):
    #         if i > 100 and np.average(self.rewards[-100:]) >= 100:
    #             print("Done training")
    #             return

    #         for _ in range(n_episodes):
    #             reward = self.run(
    #                 environment=environment,
    #                 seed=seed,
    #                 steps_limit=steps_limit
    #             )
                
    #             self.rewards.append(reward)

    #             self.train_batch(
    #                 gamma=gamma,
    #                 memory_batch_size=memory_batch_size,
    #                 batch_size=batch_size,
    #                 loss_fn=loss_fn,
    #                 optimizer=optimizer
    #             )
            
    #         self.policy.decay_epsilon()

    def train(
        self, 
        environment: gym.Env,
        n_episodes: int,
        gamma: float, 
        memory_batch_size: int,
        batch_size: int,
        steps_limit:int,
        loss_fn: nn.Module=nn.CrossEntropyLoss(),
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
                ) 
                
            self.rewards.append(total_reward)
            self.policy.decay_epsilon()
            pbar.set_postfix({'current reward': self.rewards[-1], "current epsilon": self.policy.epsilon})


    def run(self, environment: gym.Env, seed: int=42, steps_limit:int=float("inf"))-> float:
        """
        Runs a single instance of the simulation.

        @param environment: the environment
        @param seed: random seed for the environment initialisation
        """
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
        
        return total_reward
    
    def plot(self, sub_heading: str='')-> None:
        plt.plot(self.rewards)
        x = list(range(len(self.rewards)))
        z = np.polyfit(x, self.rewards, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x))
        plt.plot(list(range(-200, 200, round(400 / len(self.rewards)))))
        plt.title(f"Total reward for run per episode per iteration.\n{sub_heading}")
        plt.xlabel("episode per iteration.")
        plt.ylabel("Total reward for run.")
        plt.show()
