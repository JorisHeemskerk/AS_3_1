import torch
from torch import nn
from tqdm import tqdm


class QNetwork(nn.Module):
    """
    QNetwork class.

    Extends nn.Module class
    """
    def __init__(self)-> None:
        """
        Initializer for QNetwork.
        """
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8, 150),
            nn.ReLU(),
            nn.Linear(150, 120),
            nn.ReLU(),
            nn.Linear(120, 4),
        )
        self.device = (
            "cuda"if torch.cuda.is_available()
            else 
                "mps" if torch.backends.mps.is_available()
            else 
                "cpu"
        )
        self.to(self.device)
        print(f"Using \033[32m{self.device }\033[0m device\n")

    def forward(self, state: torch.Tensor)-> torch.Tensor:
        """
        Gets the output from the model.

        @param state: input for the model
        """
        state = state.to(self.device)
        logits = self.linear_relu_stack(state)
        return logits
    
    def train_model(
        self, 
        train_loader: torch.utils.data.DataLoader, 
        loss_fn: nn.Module, 
        optimizer: torch.optim
        )-> None:
        """
        Trains the model.

        @param train_loader: trainingsdata
        @param loss_fn: loss function
        @param optimizer: optimizer function
        """
        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)
            
            pred = self.forward(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()

            loss.backward()
            
            optimizer.step()
