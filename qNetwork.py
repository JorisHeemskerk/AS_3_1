import torch
from torch import nn


class QNetwork(nn.Module):
    """
    QNetwork class.

    Extends nn.Module class
    """
    def __init__(self, device:str=None)-> None:
        """
        Initializer for QNetwork.
        """
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 150),
            nn.ReLU(),
            nn.Linear(150, 120),
            nn.ReLU(),
            nn.Linear(120, 4),
        )
        self.device = device
        if not device:
            self.device = (
                "cuda"if torch.cuda.is_available()
                else 
                    "mps" if torch.backends.mps.is_available()
                else 
                    "cpu"
            )
        self.to(self.device)
        print(f"Using \033[32m{self.device }\033[0m device\n")

    def forward(self, data: torch.Tensor)-> torch.Tensor:
        """
        Gets the output from the model.

        @param data: input for the model
        """
        return self.linear_relu_stack(data)
    
    def train_model(
        self, 
        X: torch.tensor, 
        Y: torch.tensor, 
        loss_fn: nn.Module, 
        optimizer: torch.optim
        )-> None:
        """
        Trains the model.

        @param train_loader: trainingsdata
        @param loss_fn: loss function
        @param optimizer: optimizer function
        """
        loss = loss_fn(X, Y)
        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()

    def load(self, filename: str)-> None:
        """
        Loads network from filename

        @param filename: desired filename (+ path) of the model
        """
        self.load_state_dict(torch.load(filename))

    def save(self, filename: str)-> None:
        """
        saves network to filename

        @param filename: desired filename (+ path)
        """
        torch.save(self.state_dict(), filename)