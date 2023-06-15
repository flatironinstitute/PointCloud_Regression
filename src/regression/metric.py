import numpy as np
import torch
from enum import Enum
from abc import ABC, abstractmethod
import regression.adj_util as A
import regression.config as cf
import regression.penalties as P

###Factory Pattern of the loss functions###
class Loss(Enum):
    frobenius, chordal_quat, chordal_amat, six_d = 1, 2, 3, 4 

class LossFn(ABC):
    @abstractmethod   
    def compute_loss(self, predict: torch.Tensor, q_target: torch.Tensor,
                        config: cf.PointNetTrainConfig) -> torch.Tensor:
        pass

class FrobneiusLoss(LossFn):
    def compute_loss(self, predict: torch.Tensor, q_target: torch.Tensor,
                        config: cf.PointNetTrainConfig) -> torch.Tensor:
        adj_pred = A.vec_to_adj(predict)
        adj_quat = A.batch_quat_to_adj(q_target)
        loss = frobenius_norm_loss(adj_pred, adj_quat)
        if config.constrain:
            norm_penalty = P.penalty_sum(predict,config.select_constrain)
            loss = loss + config.cnstr_pre*norm_penalty
        q_pred = A.batch_adj_to_quat(adj_pred)
        return loss, q_pred
    #Singleton pattern
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FrobneiusLoss, cls).__new__(cls)
        return cls.instance

class AMatirxLoss(LossFn):
    def compute_loss(self, predict: torch.Tensor, q_target: torch.Tensor,
                        config:cf.PointNetTrainConfig) -> torch.Tensor:
        anti_quat = A.vec_to_quat(predict)
        loss = chordal_square_loss(anti_quat, q_target)
        return loss, anti_quat
    #Singleton pattern
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(AMatirxLoss, cls).__new__(cls)
        return cls.instance

class ChordalLoss(LossFn):
    def compute_loss(self, predict: torch.Tensor, q_target: torch.Tensor, 
                        config: cf.PointNetTrainConfig) -> torch.Tensor:
        loss = chordal_square_loss(predict, q_target)
        return loss, predict
    #Singleton pattern
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ChordalLoss, cls).__new__(cls)
        return cls.instance

class SixDLoss(LossFn):
    def compute_loss(self, predict: torch.Tensor, q_target: torch.Tensor, 
                        config: cf.PointNetTrainConfig) -> torch.Tensor:
        rot_mat = A.sixdim_to_rotmat(predict)
        mat_quat = A.quat_to_rotmat(q_target)
        loss = frobenius_norm_loss(rot_mat, mat_quat)
        q_pred = A.rotmat_to_quat(rot_mat)
        return loss, q_pred
    #Singleton pattern
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(SixDLoss, cls).__new__(cls)
        return cls.instance

class LossFactory:
    def create(self, loss_name):
        switcher = {
            'adjugate': FrobneiusLoss(),
            'a-matrix': AMatirxLoss(),
            'chordal': ChordalLoss(),
            'six-d': SixDLoss()
        }
        return switcher.get(loss_name)

###helper functions of loss, and angle differences###
def quat_cosine_diff(q_a: torch.Tensor, q_b: torch.Tensor) -> torch.Tensor:
    """
    Calculate batch of quaternion cosine differences
    """
    assert q_a.shape == q_b.shape
    assert q_a.shape[-1] == q_b.shape[-1]

    batch_dot = torch.bmm(q_a.unsqueeze(1), q_b.unsqueeze(2)).squeeze()
    batch_q_diff = torch.abs(batch_dot)

    return batch_q_diff

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
    losses = 2*dist*dist*(4 - dist*dist)
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

