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
from IPython import display


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
        self.mean_rewards = []

    def train_batch(
        self, 
        gamma: float, 
        memory_batch_size: int,
        loss_fn: nn.Module=nn.CrossEntropyLoss(),
        )-> None:
        """
        Train the policy on a batch of transitions.

        @param gamma: discount value
        @param memory_batch_size: number of samples from memory
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
            ) if not transition.terminated else 0

            current_state_q_values = self.policy.forward(
                transition.state
            )
            train_Xs.append(torch.clone(current_state_q_values))

            current_state_q_values[
                transition.action.value
            ] = transition.reward if transition.terminated else \
            transition.reward + gamma * best_value
            
            train_ys.append(current_state_q_values)

        self.policy.train(
            X_train=train_Xs,
            y_train=train_ys,
            loss_fn=loss_fn,
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
        Train the policy.

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
        pbar = tqdm(range(n_episodes))
        fig, ax = plt.subplots(ncols=2, figsize=(10,5))
        dh = display.display(fig, display_id=True)
        for i in pbar:
            if i > n_episodes_to_average and np.mean(self.rewards[-n_episodes_to_average:]) >= threshold_stop_condition:
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
                step_number += 1 
                self.train_batch(
                    gamma=gamma,
                    memory_batch_size=memory_batch_size,
                    loss_fn=loss_fn,
                ) 
                self.policy.decay_epsilon()
                
            self.rewards.append(total_reward)
            self.mean_rewards.append(np.mean(self.rewards[-n_episodes_to_average:]))

            # tqdm debug
            pbar.set_postfix({
                f"\033[{'31' if self.mean_rewards[-1] < 0 else '32'}mlast {n_episodes_to_average} eps R avg'" : 
                    f"\033[1;{'31' if self.mean_rewards[-1] < 0 else '32'}m{self.mean_rewards[-1]}\033[0;37m",
                f"\033[{'31' if self.rewards[-1] < 0 else '32'}m{'R'}" : 
                    f"\033[1;{'31' if self.rewards[-1] < 0 else '32'}m{self.rewards[-1]}\033[0;37m", 
                "\033[0;36mε" : f"\033[1;36m{self.policy.epsilon}\033[0;37m", 
                "\033[0;35mmem sz" : f"\033[1;35m{str(self.memory)}\033[0;37m",
                f"last {n_episodes_to_average} eps Rs" : f"\033[1;37m{[int(reward) for reward in self.rewards[-n_episodes_to_average:]]}\033[0;37m"
            })
            if i > 1:
                ax[0].cla()
                ax[1].cla()
                self.plot(sub_heading=f"{n_episodes} eps, {memory_batch_size} Mbsz, {self.policy.decay} dec", show=False, fig= fig, ax=ax)
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
    
    def plot(self, sub_heading: str='', show: bool=True, fig: plt.Figure=None, ax=None)-> None:
        if not fig and not ax:
            fig, ax = plt.subplots(ncols=2, figsize=(10,5))
        assert fig is not None and ax is not None, "forgot to pass fig or ax."

        fig.suptitle(sub_heading, size=16, color="purple")

        ax[0].plot(self.rewards)
        x = list(range(len(self.rewards)))
        z = np.polyfit(x, self.rewards, 1)
        p = np.poly1d(z)
        ax[0].plot(x, p(x))
        ax[0].plot(list(range(-200, 200, int(np.ceil(400 / len(self.rewards))))))
        ax[0].set_title(f"Total reward for run per iteration.")
        ax[0].set_xlabel("iteration")
        ax[0].set_ylabel("total reward for run")
        

        ax[1].plot(self.mean_rewards)
        ax[1].set_title(f"Mean reward for run per iteration.")
        ax[1].set_xlabel("iteration")
        ax[1].set_ylabel("mean reward for run")

        if show:
            fig.show()