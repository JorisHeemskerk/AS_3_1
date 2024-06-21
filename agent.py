from memory import Memory
from policy import Policy
from transition import Transition
from state import State

import torch
from torch import nn
import gymnasium as gym


class Agent:
    """
    Agent class.
    """

    def __init__(self, policy: Policy, memory: Memory)-> None:
        self.policy = policy
        self.memory = memory

    def train_batch(
        self, 
        gamma: float, 
        batch_size: int,
        num_epochs: int,
        loss_fn: nn.Module=nn.CrossEntropyLoss(),
        optimizer: torch.optim.Optimizer=torch.optim.Adam,
        n_episodes: int=64
        )-> None:
        assert n_episodes > 0, \
            "Bruh, what did you expect to happen? (╯‵□′)╯︵┻━┻"

        batch: list[Transition] = self.memory.get_batch(n_episodes)

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
            num_epochs=num_epochs,
            loss_fn=loss_fn,
            optimizer=optimizer
        )

    def train(
        self, 
        environment: gym.Env,
        n_iterations: int,
        gamma: float, 
        batch_size: int,
        num_epochs: int,
        loss_fn: nn.Module=nn.CrossEntropyLoss(),
        optimizer: torch.optim.Optimizer=torch.optim.Adam,
        n_episodes: int=64,
        seed: int=42,
        )-> None:

        for i in range(n_iterations):
            print(f"running iteration {i}")
            while not self.memory.is_full():
                self.run(environment=environment, seed=seed)

            self.train_batch(
                gamma=gamma, 
                batch_size=batch_size,
                num_epochs=num_epochs,
                loss_fn=loss_fn,
                optimizer=optimizer,
                n_episodes=n_episodes
            )

            self.memory.clear()
            

    def run(self, environment: gym.Env, seed: int=42)-> None:
        """
        """
        # print("Welcome to the run function")
        start_state, _ = environment.reset(seed=seed)
        start_state = State(*start_state)
        while True:
            action = self.policy.select_action(start_state)
            state_prime, reward, is_terminated, truncated, _ = \
                environment.step(action.value)
            state_prime = State(*state_prime)

            # print(f"trying to store to memory. Currently {self.memory._Memory__idx+1}/{self.memory.max_size} items stored")
            if not self.memory.is_full():
                # print("actually storing to memory")
                self.memory.store(
                    Transition(
                        start_state, 
                        action,
                        reward,
                        state_prime,
                        is_terminated
                    )
                )
            # print(f"checking if terminated. {is_terminated = }, {truncated = }")
            if is_terminated or truncated:
                environment.reset(seed=seed)
                break

            start_state = state_prime
