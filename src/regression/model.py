import torch
import torch.nn as nn
from typing import Optional

class PointNet(nn.Module):
    def __init__(self, hidden_size:int, num_points:int, adj_option:str, batch_norm:bool) -> None:
        """
        num_points: number of 3d points in each cloud
        adj_option: whether to train it with adjugate matrix
        """
        super().__init__()
        self.out_dim = 10
        if adj_option == "six-d":
            self.out_dim = 6 
        elif adj_option == "chordal" or adj_option == "l2chordal":
            self.out_dim = 4
            
        self.feat_net = PointFeatCNN(hidden_size, batch_norm) #feature net directly output dim of hidden layer
        self.hidden_mlp = nn.Sequential(
                                    nn.Linear(hidden_size, 256),
                                    nn.LeakyReLU(),
                                    torch.nn.Linear(256, 128),
                                    torch.nn.LeakyReLU(),
                                    torch.nn.Linear(128, self.out_dim)
                                    )
        self.soft_max = nn.Softmax()

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

class Regress2DNet(nn.Module):
    def __init__(self, n_class:int, regress_dim:int) -> None:
        super().__init__()
        self.basic_model = MobileNet(3)
        self.head_a = RegressHead(1024, n_class, regress_dim)
        self.head_e = RegressHead(1024, n_class, regress_dim)
        self.head_t = RegressHead(1024, n_class, regress_dim)
        


class MobileNet(nn.Module):
    def __init__(self, channel_in:int) -> None:
        """@args
        channel_in: number of channels of the input image
        @notice: here we keep the backbone of MobileNet, but 
        retain the output layer to RegressHead for a special task
        """
        super().__init__()
        self.model = nn.Sequential(
            ConvBatchNorm(channel_in, 32, 2),
            ConvDepthWise(32, 64, 1),
            ConvDepthWise(64, 128, 2),
            ConvDepthWise(128, 128, 1),
            ConvDepthWise(128, 256, 2),
            ConvDepthWise(256, 256, 1),
            ConvDepthWise(256, 512, 2),
            ConvDepthWise(512, 512, 1),
            ConvDepthWise(512, 512, 1),
            ConvDepthWise(512, 512, 1),
            ConvDepthWise(512, 512, 1),
            ConvDepthWise(512, 512, 1),
            ConvDepthWise(512, 1024, 2),
            ConvDepthWise(1024, 1024, 1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x) -> torch.Tensor:
        x = self.model(x)
        x = x.view(-1, 1024)
        return x

class RegressHead(nn.Module):
    def __init__(self, input_size:int, n_class:int, regress_dim:int) -> None:
        """@args
        n_class: number of classes/categories
        regress_dim: the dimension of output to be regressed; 
        can be 2 for [cos, sin] or 3 for [a^2, b^2, ab] 
        """
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, n_class*regress_dim)
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.head(x)

class ConvDepthWise(torch.nn.Module):
    def __init__(self, input:int, output:int, stride:int) ->torch.Tensor:
        super().__init__()
        self.depth_wise = nn.Sequential(
                            nn.Conv2d(input, input, 3, stride, 1, groups=input, bias=False),
                            nn.BatchNorm2d(input),
                            nn.ReLU(inplace=True),

                            nn.Conv2d(input, output, 1, 1, 0, bias=False),
                            nn.BatchNorm2d(output),
                            nn.ReLU(inplace=True)
                        )
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.depth_wise(x)
        return x

class ConvBatchNorm(torch.nn.Module):
    def __init__(self, input:int, output:int, stride:int) -> torch.Tensor:
        super().__init__
        self.conv_batch = nn.Sequential(
            nn.Conv2d(input, output, 3, stride, 1, bias=True),
            nn.BatchNorm2d(output),
            nn.ReLU(inplace=True)
        )
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.conv_batch(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, num_layer:int, hidden_size:int, out_opt:str) -> None:
        super().__init__()
        
        if out_opt == "adjugate":
            out = 10
        else:
            out = 4

        self.relu = nn.ReLU()
        self.input_layer = nn.Linear(9,hidden_size)
        
        layers = []
        for i in range(num_layer):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*layers)

        self.output_layer = nn.Linear(hidden_size,out)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x_1 = x[:, 0, :, :].transpose(1,2) #should in dim bx3xnum
        x_2 = x[:, 1, :, :] #should in dim bxnumx3
        batch = len(x)

        cov_mat = torch.bmm(x_1, x_2).view(batch, -1)
        
        x_ = self.input_layer(cov_mat)
        x_ = self.hidden(x_)
        x_ = self.output_layer(x_)

        return x_
