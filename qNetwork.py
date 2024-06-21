import torch
from torch import nn
from tqdm import tqdm


class QNetwork(nn.Module):
    def __init__(self):
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

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        x = x.to(self.device)
        # print(f"{x = }, {type(x) = }")
        logits = self.linear_relu_stack(x)
        # print(f"{logits = }, {type(logits) = }")
        return logits
    
    def train_model(
        self, 
        train_loader: torch.utils.data.DataLoader, 
        loss_fn: nn.Module, 
        optimizer: torch.optim, 
        num_epochs: int
    ):
        for epoch in range(num_epochs):
            print(f"-------------------------------\nEpoch {epoch+1}")
            for batch, (X, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
                X, y = X.to(self.device), y.to(self.device)
                
                # Compute prediction and loss
                pred = self.forward(X)
                loss = loss_fn(pred, y)

                # Backpropagation
                loss.backward(retain_graph=True)
                optimizer.zero_grad()
                optimizer.step()

                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
        print("Training done!")