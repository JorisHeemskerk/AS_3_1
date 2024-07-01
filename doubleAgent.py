from memory import Memory
from transition import Transition
from agent import Agent

import torch
from torch import nn
from dataclasses import astuple


class DoubleAgent(Agent):
    """
    double deep q-learning agent
    """

    def __init__(
        self, 
        network: nn.Module, 
        target_network: nn.Module, 
        memory: Memory, 
        optimizer: torch.optim.Optimizer,
        epsilon: float,
        decay: float,
        n_actions: int,
        tau: float
    ) -> None:
        super().__init__(network, memory, optimizer, epsilon, decay, n_actions)

        self._target_network = target_network
        self.tau = tau        

    def train_batch(
        self, 
        gamma: float, 
        memory_batch_size: int,
        loss_fn: nn.Module=nn.MSELoss(),
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
        next_q_values = self._target_network.forward(next_states).max(1)[0].detach() 
        
        expected_q_values = rewards + (gamma * next_q_values * (1 - terminateds))

        self._network.train_model(
            X=current_q_values, 
            Y=expected_q_values.unsqueeze(1), 
            loss_fn=loss_fn,
            optimizer=self.optimizer
        )

        self.align_target_network()        

    def align_target_network(self):
        for param, target_param in zip(
            self._network.parameters(), 
            self._target_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * 
                param.data + 
                target_param.data * 
                (1.0 - self.tau)
            )