from memory import Memory
from transition import Transition

import torch
from torch import nn
import gymnasium as gym
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
import random
from dataclasses import astuple


class Agent:
    """
    Agent class.
    """

    def __init__(
        self, 
        network: nn.Module, 
        memory: Memory, 
        optimizer: torch.optim.Optimizer,
        epsilon: float,
        decay: float,
        n_actions: int
    )-> None:
        """
        Initializer for Agent.

        @param memory: Agent's memory
        """
        self._network = network
        self.memory = memory
        self.optimizer = optimizer(self._network.parameters(), lr=0.001)
        self.epsilon = epsilon
        self.decay = decay
        self.rewards = []
        self.mean_rewards = []
        self.n_actions = n_actions

    def train_batch(
        self, 
        gamma: float, 
        memory_batch_size: int,
        loss_fn: nn.Module=nn.CrossEntropyLoss(),
        )-> None:
        """
        Train the Q-network on a batch of transitions.

        @param gamma: discount value
        @param memory_batch_size: number of samples from memory
        @param loss_fn: loss function, default=nn.CrossEntropyLoss
        """
        batch: list[Transition] = self.memory.get_batch(
            batch_size=memory_batch_size
        )
        batch = Transition(*zip(*batch))

        states = torch.FloatTensor(batch.state).to(self._network.device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self._network.device)
        rewards = torch.FloatTensor(batch.reward).to(self._network.device)
        next_states = torch.FloatTensor(batch.next_state).to(self._network.device)
        terminateds = torch.FloatTensor(batch.terminated).to(self._network.device)

        current_q_values = self._network.forward(states).gather(1, actions)

        # Detach makes sure no gradients are calculated
        next_q_values = self._network.forward(next_states).max(1)[0].detach() 
        
        expected_q_values = rewards + (gamma * next_q_values * (1 - terminateds))

        self._network.train_model(
            X=current_q_values, 
            Y=expected_q_values.unsqueeze(1), 
            loss_fn=loss_fn,
            optimizer=self.optimizer
        )

    def train(
        self, 
        environment: gym.Env,
        n_episodes: int,
        n_episodes_to_average: int,
        threshold_stop_condition: int,
        gamma: float, 
        memory_batch_size: int,
        steps_limit:int,
        loss_fn: nn.Module=nn.MSELoss(),
        seed: int=42,
        )-> None:
        """
        Train the Q-Network

        @param environment: the environment
        @param n_iterations: number of iterations
        @param gamma: discount value
        @param memory_batch_size: number of samples from memory
        @param loss_fn: loss function, default=nn.CrossEntropyLoss
        @param optimizer: optimizer function, default=torch.optim.Adam
        @param n_episodes: number of episodes
        @param seed: random seed for the environment initialisation
        """
        self.rewards = []
        self.mean_rewards = []

        fig, ax = plt.subplots(ncols=2, figsize=(10,5))
        dh = display.display(fig, display_id=True)

        pbar = tqdm(range(n_episodes))
        for i in pbar:
            if i > n_episodes_to_average and self.mean_rewards[-1] >= threshold_stop_condition:
                print("Done training, it good enough d=====(￣▽￣*)b")
                plt.close()
                return
            total_reward = 0
            start_state, _ = environment.reset(seed=seed)
            for _ in range(steps_limit):
                
                action = self.select_action(start_state)

                state_prime, reward, is_terminated, truncated, _ = \
                    environment.step(action)

                total_reward += reward
                
                self.memory.store(
                    Transition(
                        list(start_state), 
                        action,
                        reward,
                        list(state_prime),
                        is_terminated
                    )
                )

                if is_terminated or truncated:
                    break

                start_state = state_prime
                if len(self.memory) > memory_batch_size:
                    self.train_batch(
                        gamma=gamma,
                        memory_batch_size=memory_batch_size,
                        loss_fn=loss_fn,
                    ) 
                self.decay_epsilon()
                
            self.rewards.append(total_reward)
            self.mean_rewards.append(np.mean(self.rewards[-n_episodes_to_average:]))

            # tqdm debug
            pbar.set_postfix({
                f"\033[{'31' if self.mean_rewards[-1] < 0 else '32'}mlast {n_episodes_to_average} eps R avg" : 
                    f"\033[1;{'31' if self.mean_rewards[-1] < 0 else '32'}m{int(self.mean_rewards[-1])}\033[0;37m",
                f"\033[{'31' if self.rewards[-1] < 0 else '32'}m{'R'}" : 
                    f"\033[1;{'31' if self.rewards[-1] < 0 else '32'}m{int(self.rewards[-1])}\033[0;37m", 
                "\033[0;36mε" : f"\033[1;36m{round(self.epsilon, 2)}\033[0;37m", 
                "\033[0;35mmem sz" : f"\033[1;35m{str(self.memory)}\033[0;37m",
            })
            if i > 1 and i % 10 == 0:
                ax[0].cla()
                ax[1].cla()
                self.plot(sub_heading=f"{n_episodes} eps, {memory_batch_size} Mbsz, {self.decay} dec", show=False, fig= fig, ax=ax)
                dh.update(fig)
        plt.close()

    def run(self, environment: gym.Env, seed: int=42, steps_limit:int=float("inf"))-> float:
        """
        Runs a single instance of the simulation.

        @param environment: the environment
        @param seed: random seed for the environment initialisation
        """
        total_reward = 0
        start_state, _ = environment.reset(seed=seed)
        step_number = 0
        while step_number < steps_limit:
            action = self.select_action(start_state)
            state_prime, reward, is_terminated, truncated, _ = \
                environment.step(action)

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
    
    def decay_epsilon(self)-> None:
        """
        Decrease the epsilon by mulitplying it with a constant. *e^0.005
        """
        self.epsilon *= self.decay
        if self.epsilon <= 0.01:
            self.epsilon = 0.01

    def select_action(self, state: np.ndarray)-> int:
        """
        Select action based on current Q-network.
        
        @param state: Current state to perform action in.

        @return int with index of action to perform.
        """
        if random.random() < self.epsilon:
            return random.choice(range(self.n_actions))
        with torch.no_grad():
            return torch.argmax(self._network.forward(
                data=torch.FloatTensor(state).unsqueeze(0).to(self._network.device)
            )).item()

    def plot(
        self, 
        sub_heading: str='', 
        show: bool=True, 
        fig: plt.Figure=None, 
        ax=None
    )-> None:
        if fig is None or ax is None:
            fig, ax = plt.subplots(ncols=2, figsize=(10,5))
        assert fig is not None and ax is not None, "forgot to pass fig or ax."

        fig.suptitle(sub_heading, size=16, color="purple")

        ax[0].plot(self.rewards)
        x = list(range(len(self.rewards)))
        z = np.polyfit(x, self.rewards, 1)
        p = np.poly1d(z)
        ax[0].plot(x, p(x))
        ax[0].set_title(f"Total reward for run per iteration.")
        ax[0].set_xlabel("iteration")
        ax[0].set_ylabel("total reward for run")
        

        ax[1].plot(self.mean_rewards)
        ax[1].set_title(f"Mean reward for run per iteration.")
        ax[1].set_xlabel("iteration")
        ax[1].set_ylabel("mean reward for run")

        if show:
            fig.show()
