import numpy as np
import torch

def quat_norm_diff(q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
    """
    calculate batch of quaternion diff
    """
    assert(q_a.shape == q_b.shape)
    assert(q_a.shape[-1] == 4)

    return torch.min((q_a - q_b).norm(dim = 1), (q_a + q_b).norm(dim = 1)).squeeze()

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