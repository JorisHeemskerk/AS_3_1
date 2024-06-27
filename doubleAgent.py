from memory import Memory
from policy import Policy
from transition import Transition
from agent import Agent

import torch
from torch import nn


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
        loss_fn: nn.Module=nn.MSELoss(),
    )-> None:
        if self.memory._size <= memory_batch_size:
            return
        
        batch: list[Transition] = self.memory.get_batch(
            batch_size=memory_batch_size
        )

        train_Xs = []
        train_ys = []

        for transition in batch:
            next_state_q_values = self.target_policy._network.forward(
                torch.Tensor(tuple(transition.next_state))
            )
        
            best_value = max(
                next_state_q_values
            ) if not transition.terminated else 0

            current_state_q_values = self.policy._network.forward(
                torch.Tensor(tuple(transition.state))
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
        
        self.align_target_network()

        self.policy.decay_epsilon()
        self.target_policy.decay_epsilon()

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