import torch.nn.functional as F
from torch import nn
import torch


class BBandFChead(nn.Module):
    def __init__(self, bb_model, train_bb=False, hidden_dim=512, output_dim=2):
        super().__init__()
        self.train_bb = train_bb
        self.bb_model = bb_model
        self.flatten = nn.Flatten()
        if hidden_dim > 0:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(2048, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.linear_relu_stack = nn.Linear(2048, output_dim)

    def forward(self, x):
        if not self.train_bb:
            with torch.no_grad():
                x = self.bb_model(x)
        else:
            x = self.bb_model(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits