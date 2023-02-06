import numpy as np
import torch

def quat_norm_diff(q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
    assert(q_a.shape == q_b.shape)
    assert(q_a.shape[-1] == 4)

    return torch.min((q_a - q_b).norm(dim = 1), (q_a + q_b).norm(dim = 1)).squeeze()

def chordal_square_loss(q_predict: torch.Tensor, q_target: torch.Tensor, reduce = True) -> torch.Tensor:
    assert(q_predict.shape == q_target.shape)

    dist = quat_norm_diff(q_predict, q_target)
    losses = 2*dist**2*(4 - dist**2)
    loss = losses.mean() if reduce else losses

    return loss

def mean_square(q_predict: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    mse = torch.nn.MSELoss()
    return mse(q_predict, q_target)