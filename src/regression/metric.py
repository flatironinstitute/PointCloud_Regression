import numpy as np
import torch
from enum import Enum
from abc import ABC, abstractmethod

class Loss(Enum):
    frobenius, chordal_quat, chordal_amat, six_d = 1, 2, 3, 4 

class LossFn(ABC):
    @abstractmethod   
    def compute_loss(self, q_predict: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
        pass

def quat_norm_diff(q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
    """
    Calculate batch of quaternion differences.
    """
    assert q_a.shape == q_b.shape
    assert q_a.shape[-1] == 4

    diff = (q_a - q_b).norm(dim=1)
    sum_ = (q_a + q_b).norm(dim=1)
    out = torch.min(diff, sum_).squeeze()

    # Ensure output tensor is on the same device as inputs
    if q_a.device != q_b.device:
        out = out.to(q_a.device)

    return out

def quat_norm_to_angle(q_norms: torch.Tensor, units='deg'):
    """
    convert list of delta_q to angle
    """
    angle = 4.*torch.asin(0.5*q_norms)
    if units == 'deg':
        angle = (180./np.pi)*angle
    elif units == 'rad':
        pass
    else:
        raise RuntimeError('Unknown units in metric conversion.')
    
    return angle

def quat_angle_diff(q_a: torch.Tensor, q_b: torch.Tensor, units='deg', reduce=True):
    """
    get angle diff for a batch of quaternions
    """

    assert(q_a.shape == q_b.shape)
    assert(q_a.shape[-1] == 4)
    diffs = quat_norm_to_angle(quat_norm_diff(q_a, q_b), units=units)
    return diffs.mean() if reduce else diffs

def chordal_square_loss(q_predict: torch.Tensor, q_target: torch.Tensor, reduce = True) -> torch.Tensor:
    #usually works well for a normalized cloud
    assert(q_predict.shape == q_target.shape)

    dist = quat_norm_diff(q_predict, q_target)
    losses = 2*dist**2*(4 - dist**2)
    loss = losses.mean() if reduce else losses

    return loss

def mean_square(q_predict: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    mse = torch.nn.MSELoss()
    return mse(q_predict, q_target)

def frobenius_norm_loss(adj_mat_src: torch.Tensor, adj_mat_trg: torch.Tensor, reduce = True) -> torch.Tensor:
    """
    get the Frobenius norm of batches of rotation between
    source and the target adjugate matrix
    both should be batches of 4x4 matrix
    """
    print("shape of predicted adj_src: ", adj_mat_src.shape)
    print("shape of g.t. adj_trg: ", adj_mat_trg.shape)
    assert(adj_mat_src.shape == adj_mat_trg.shape)

    if adj_mat_src.dim() < 3:
        adj_mat_src.unsqueeze(dim = 0)
        adj_mat_trg.unsqueeze(dim = 0)
    losses = (adj_mat_src - adj_mat_trg).norm(dim = [1,2])
    loss = losses.mean() if reduce else losses
    return loss

