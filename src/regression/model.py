import torch
import torch.nn as nn
from typing import Optional

class PointNet(nn.Module):
    def __init__(self, hidden_size:int, num_points:int, adj_option:bool, batch_norm:bool) -> None:
        """
        num_points: number of 3d points in each cloud
        adj_option: whether to train it with adjugate matrix
        """
        super().__init__()
        self.in_channel = 3*num_points*2
        self.out_dim = 4 if not adj_option else 10
        self.feat_net = PointFeatCNN(hidden_size, batch_norm) #feature net directly output dim of hidden layer
        self.hidden_mlp = nn.Sequential(
                                    nn.Linear(hidden_size, 256),
                                    nn.LeakyReLU(),
                                    torch.nn.Linear(256, 128),
                                    torch.nn.LeakyReLU(),
                                    torch.nn.Linear(128, self.out_dim)
                                    )

    def forward(self, x) -> torch.Tensor:
        x_1 = x[:, 0, :, :].transpose(1,2)
        x_2 = x[:, 1, :, :].transpose(1,2)

        x = self.feat_net(torch.cat([x_1, x_2], dim=1))
        x = self.hidden_mlp(x)

        return x

class PointFeatCNN(nn.Module):
    def __init__(self, feature_dim:int, batch_norm=False):
        super().__init__()
        if batch_norm:
            self.net = nn.Sequential(
                            nn.Conv1d(6, 64, kernel_size=1),
                            nn.BatchNorm1d(64),
                            nn.LeakyReLU(),
                            nn.Conv1d(64, 128, kernel_size=1),
                            nn.BatchNorm1d(128),
                            nn.LeakyReLU(),
                            nn.Conv1d(128, feature_dim, kernel_size=1),
                            nn.AdaptiveMaxPool1d(output_size=1)
                    )
        else:
            self.net = nn.Sequential(
                        nn.Conv1d(6, 64, kernel_size=1),
                        nn.LeakyReLU(),
                        nn.Conv1d(64, 128, kernel_size=1),
                        nn.LeakyReLU(),
                        nn.Conv1d(128, 1024, kernel_size=1),
                        nn.AdaptiveMaxPool1d(output_size=1)
                    )
    def forward(self, x):
        x = self.net(x)
        return x.squeeze()
