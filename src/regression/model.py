import torch
import torch.nn as nn
from typing import Optional

class FeedForward(nn.Module):
    def __init__(self, num_hidden:int, hidden_size:int, num_points:int) -> None:
        super().__init__()
        self.in_channel = 3*num_points*2
        
        self.input_layer = nn.Linear(self.in_channel,hidden_size)
        self.hidden_mlp = nn.Sequential(nn.Linear(hidden_size,hidden_size),
                            nn.LeakyReLU())
        self.output_layer = nn.Linear(hidden_size, 4)
        self.num_hidden = num_hidden

    def forward(self, x) -> torch.Tensor:
        x = self.input_layer(x)

        for i in range(self.num_hidden):
            x = self.hidden_mlp(x)
        x = self.output_layer(x)

        return x
