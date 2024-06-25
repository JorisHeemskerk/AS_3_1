from action import Action
from state import State

import torch
from torch import nn
from dataclasses import astuple
import random


class Policy:
    """
    Policy class
    """

    def __init__(
        self, 
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        epsilon: float,
        decay: float
    )-> None:
        """
        Initializer for Policy.

        @param network: Pytorch Neural Network
        @param epsilon: Epsilon, what did you expect...
        """
        self._network = network
        self.optimizer = optimizer(self._network.parameters(), lr=0.001)
        self.epsilon = epsilon
        self.decay = decay

    def load(self, filename: str)-> None:
        """
        Loads network from filename

        @param filename: desired filename (+ path) of the model
        """
        self._network.load_state_dict(torch.load(filename))

    def save(self, filename: str)-> None:
        """
        saves network to filename

        @param filename: desired filename (+ path)
        """
        torch.save(self._network.state_dict(), filename)

    def forward(self, state: State)-> torch.Tensor:
        state_to_pass = torch.tensor(astuple(state), dtype=torch.float32).unsqueeze(0)
        
        return self._network.forward(
            state_to_pass
        ).squeeze(0)
    
    def decay_epsilon(self)-> None:
        """
        Decrease the epsilon by mulitplying it with a constant.
        """
        self.epsilon *= self.decay
        if self.epsilon <= 0.01:
            self.epsilon = 0.01

    def train(
        self, 
        X_train: list[State],
        y_train: list[torch.Tensor],
        batch_size: int,
        loss_fn: nn.Module=nn.CrossEntropyLoss(),
    )-> None:
        """
        Train the Q network on a prepared memory batch.

        @param X_train: list of State object to train
        @param y_train: list of labels. 
        @param batch_size: batch size for training the model
        @param num_epochs: number of epochs for training the model
        @param loss_fn: loss function, default=nn.CrossEntropyLoss
        """
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor([astuple(x) for x in X_train], dtype=torch.float32),
            torch.stack(y_train, dim=0)
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )

        self._network.train_model(
            train_loader=train_loader, 
            loss_fn=loss_fn, 
            optimizer=self.optimizer
        ) 

    def select_action(self, state: State)-> Action:
        """
        Select action based on current policy.
        
        @param state: Current State to perform action in.

        @return Action with Action to perform.
        """
        if random.random() < self.epsilon:
            return random.choice(list(Action))
        return Action(torch.argmax(self.forward(state=state)).item())
