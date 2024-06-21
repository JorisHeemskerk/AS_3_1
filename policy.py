from action import Action
from state import State

import torch
from torch import nn
from dataclasses import astuple

class Policy:
    """
    Policy class
    """

    def __init__(
        self, 
        network: nn.Module, 
        epsilon: float
    )-> None:
        """
        Initializer for Policy.

        @param network: Pytorch Neural Network
        @param epsilon: Epsilon, what did you expect...
        """
        self._network = network
        self.epsilon = epsilon

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
        state_to_pass = torch.tensor(astuple(state))
        print(f"{state_to_pass = }")
        return self._network.forward(
            state_to_pass
        )
    
    def decay()-> None:
        """
        TODO: degrade epsilon (optional)
        """
        pass

    def train(
        self, 
        X_train: list[State],
        y_train: list[torch.Tensor],
        batch_size: int,
        num_epochs: int,
        loss_fn: nn.Module=nn.CrossEntropyLoss(),
        optimizer: torch.optim.Optimizer=torch.optim.Adam
    )-> None:
        """
        Train the Q network on a prepared memory batch.

        @param X_train: list of State object to train
        @param y_train: list of labels. 
         Can be either floats or torch.tensors 
        @param batch_size: batch size for training the model
        @param num_epochs: number of ecpohs for training the model
        @param loss_fn: loss function, default=nn.CrossEntropyLoss
        @param optimizer: optimizer function, default=torch.optim.Adam
        """
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor([astuple(x) for x in X_train]),
            torch.stack(y_train, dim=0)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optimizer(self._network.parameters(), lr=0.001)

        self._network.train_model(
            train_loader=train_loader, 
            loss_fn=loss_fn, 
            optimizer=optimizer, 
            num_epochs=num_epochs
        ) 

    def select_action(self, state: State)-> Action:
        """
        Select action based on current policy.
        
        @param state: Current State to perform action in.

        @return Action with Action to perform.
        """
        a = torch.argmax(self.forward(state=state))
        print(f"in Policy::select_action1, {a = }")
        print(f"in Policy::select_action2, {a.item()}")
        print(f"in Policy::select_action2, {type(a.item())}")
        print(f"in Policy::select_action2, {Action(a.item())}")
        # exit(0)
        return Action(a.item())
