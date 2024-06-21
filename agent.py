from memory import Memory
from policy import Policy
from transition import Transition
from state import State

import torch
from torch import nn
import gymnasium as gym
from tqdm import tqdm


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

    def train_batch(
        self, 
        gamma: float, 
        batch_size: int,
        loss_fn: nn.Module=nn.CrossEntropyLoss(),
        optimizer: torch.optim.Optimizer=torch.optim.Adam
        )-> None:
        """
        Train the policy on a batch of transitions.

        @param gamma: discount value
        @param batch_size: batch size for training the model
        @param loss_fn: loss function, default=nn.CrossEntropyLoss
        @param optimizer: optimizer function, default=torch.optim.Adam 
        """
        if self.memory._Memory__idx < batch_size:
            return

        batch: list[Transition] = self.memory.get_batch(batch_size=batch_size)

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
            optimizer=optimizer
        )


    def train(
        self, 
        environment: gym.Env,
        n_iterations: int,
        gamma: float, 
        batch_size: int,
        loss_fn: nn.Module=nn.CrossEntropyLoss(),
        optimizer: torch.optim.Optimizer=torch.optim.Adam,
        n_episodes: int=64,
        seed: int=42,
        )-> None:
        """
        Train the policy.

        @param environment: the environment
        @param n_iterations: number of iterations
        @param gamma: discount value
        @param batch_size: batch size for training the model
        @param loss_fn: loss function, default=nn.CrossEntropyLoss
        @param optimizer: optimizer function, default=torch.optim.Adam
        @param n_episodes: number of episodes
        @param seed: random seed for the environment initialisation
        """
        for i in range(n_iterations):
            for _ in tqdm(range(n_episodes)):
                self.run(
                    environment=environment,
                    seed=seed
                )

                self.train_batch(
                    gamma=gamma,
                    batch_size=batch_size,
                    loss_fn=loss_fn,
                    optimizer=optimizer
                )
            
                self.policy.decay_epsilon()
            print(f"klaar met iteratie {i+1}")

            

    def run(self, environment: gym.Env, seed: int=42)-> None:
        """
        Runs a single instance of the simulation.

        @param environment: the environment
        @param seed: random seed for the environment initialisation
        """
        start_state, _ = environment.reset(seed=seed)
        start_state = State(*start_state)
        while True:
            action = self.policy.select_action(start_state)
            state_prime, reward, is_terminated, truncated, _ = \
                environment.step(action.value)
            state_prime = State(*state_prime)

            if not self.memory.is_full():
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
