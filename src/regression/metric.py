import numpy as np
import torch
from enum import Enum
from abc import ABC, abstractmethod
import scipy.linalg

import regression.adj_util as A
import regression.config as cf
import regression.penalties as P

###Factory Pattern of the loss functions###
class Loss(Enum):
    frobenius, chordal_quat, chordal_amat, six_d, rmsd = 1, 2, 3, 4, 5

class LossFn(ABC):
    @abstractmethod   
    def compute_loss(self, predict: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
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
    def compute_loss(self, predict: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
        anti_quat = A.vec_to_quat(predict)
        loss = chordal_square_loss(anti_quat, q_target)
        return loss, anti_quat
    #Singleton pattern
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(AMatirxLoss, cls).__new__(cls)
        return cls.instance

class ChordalLoss(LossFn):
    def compute_loss(self, predict: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
        loss = chordal_square_loss(predict, q_target)
        return loss, predict
    #Singleton pattern
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ChordalLoss, cls).__new__(cls)
        return cls.instance

class ChordalL2Loss(LossFn):
    def compute_loss(self, predict: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
        loss = chordal_l2_loss(predict, q_target) + P.unit_quat_constr(predict)
        return loss, predict

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ChordalL2Loss, cls).__new__(cls)
        return cls.instance

class SixDLoss(LossFn):
    def compute_loss(self, predict: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
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

class RMSDLoss(LossFn):
    """
    rmsd loss's input should be a 10 dim vector as adjugate quaternions
    """
    def compute_loss(self, predict: torch.Tensor, q_target: torch.Tensor,
                    concate_cloud: torch.Tensor, trace_norm: bool=False) -> torch.Tensor: 
        source_cloud = concate_cloud[:, 0, :, :].transpose(1,2) 
        target_cloud = concate_cloud[:, 1, :, :].transpose(1,2)
        pred_quat = A.batch_vec_to_quat(predict)
        pred_rot = A.quat_to_rotmat(pred_quat)
        rot_cloud = torch.matmul(pred_rot, source_cloud)
        
        mse = torch.nn.MSELoss()
        loss = mse(rot_cloud, target_cloud)

        return loss, pred_quat

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(RMSDLoss, cls).__new__(cls)
        return cls.instance

class LossFactory:
    def create(self, loss_name):
        switcher = {
            'adjugate': FrobneiusLoss(),
            'a-matrix': AMatirxLoss(),
            'chordal': ChordalLoss(),
            'l2chordal': ChordalL2Loss(),
            'six-d': SixDLoss(),
            'rmsd': RMSDLoss()
        }
        return switcher.get(loss_name)

###helper functions of loss, and angle differences###
def rmsd_diff(pred_quat:torch.Tensor, cloud:torch.Tensor) -> torch.Tensor:
    """Calculate batch average of RMSD difference; 
    args: quat and cloud are directly load from batch
    """
    source_cloud = cloud[:, 0, :, :].transpose(1,2) 
    target_cloud = cloud[:, 1, :, :].transpose(1,2)

    pred_rot = A.quat_to_rotmat(pred_quat)
    rot_cloud = torch.matmul(pred_rot, source_cloud)
        
    mse = torch.nn.MSELoss()
    loss = mse(rot_cloud, target_cloud)

    return loss

def quat_cosine_diff(q_a: torch.Tensor, q_b: torch.Tensor, reduce=True) -> torch.Tensor:
    """
    Calculate batch of quaternion cosine differences
    """
    assert q_a.shape == q_b.shape
    assert q_a.shape[-1] == q_b.shape[-1]

    batch_dot = torch.bmm(q_a.unsqueeze(1), q_b.unsqueeze(2)).squeeze()
    batch_q_diff = torch.acos(torch.abs(batch_dot))
    
    batch_ang_diff = 180*batch_q_diff/torch.pi

    return batch_ang_diff.mean() if reduce else batch_ang_diff

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
    # usually works well for a normalized cloud
    assert(q_predict.shape == q_target.shape)

    dist = quat_norm_diff(q_predict, q_target)
    losses = 2*dist*dist*(4 - dist*dist)
    loss = losses.mean() if reduce else losses

    return loss

def quat_norm_l2(q_predict: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    """
    Calculate batch of L2 norm quaternion.
    """
    assert(q_predict.shape == q_target.shape)

    diff_ = q_predict - q_target
    sum_ = q_predict + q_target
    diff_dot = torch.bmm(diff_.unsqueeze(1),diff_.unsqueeze(-1)).squeeze(-1).norm(dim=1)
    sum_dot = torch.bmm(sum_.unsqueeze(1),sum_.unsqueeze(-1)).squeeze(-1).norm(dim=1)
    out = torch.min(diff_dot, sum_dot).squeeze()

    if q_predict.device != q_target.device:
        out = out.to(q_predict.device)

    return out

def chordal_l2_loss(q_predict: torch.Tensor, q_target: torch.Tensor, reduce = True) -> torch.Tensor:
    """use exactly 
    """
    assert(q_predict.shape == q_target.shape)

    dist = quat_norm_l2(q_predict, q_target)
    losses = 2*dist*dist*(4 - dist*dist)
    loss = losses.mean() if reduce else losses

    return loss

def mean_square(q_predict: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    mse = torch.nn.MSELoss()
    return mse(q_predict, q_target)

def frobenius_norm_loss(mat_src: torch.Tensor, mat_trg: torch.Tensor, reduce = True) -> torch.Tensor:
    """
    get the Frobenius norm of batches of rotation between
    source and the target adjugate matrix
    both should be batches of matrix
    can be 4x4 adj or plain rot matrix
    """
    assert(mat_src.shape == mat_trg.shape)

    if mat_src.dim() < 3:
        mat_src.unsqueeze(dim = 0)
        mat_trg.unsqueeze(dim = 0)
    losses = (mat_src - mat_trg).norm(dim = [1,2])
    loss = losses.mean() if reduce else losses
    return loss

def geodesic_dist(pred_rot:torch.Tensor, gt_rot:torch.Tensor) -> torch.Tensor:
    relative_rot = torch.matmul(gt_rot.t(), pred_rot)

    # Compute the matrix logarithm by scipy
    # disp=False suppresses warnings, and the return includes an error estimate
    log_rot, err_est = scipy.linalg.logm(relative_rot.detach().cpu().numpy(), disp=False)

    frob_norm = torch.norm(torch.from_numpy(log_rot, device=pred_rot.device), p='fro')
    
    # Calculate the geodesic distance
    r_angle = frob_norm / torch.sqrt(2)
    
    return r_angle

def geodesic_batch_mean(pred_rot: torch.Tensor, gt_rot: torch.Tensor) -> torch.Tensor:
    """Calculate the mean geodesic distance for a batch of rotation matrices."""
    # Calculate geodesic distance for each item in the batch
    print("check batch dim for geodist: ", pred_rot.shape)
    batch_geodesic_distances = torch.tensor([
        geodesic_dist(pred_rot[i], gt_rot[i]) for i in range(pred_rot.shape[0])
    ], device=pred_rot.device)

    # Return the mean of the batch geodesic distances
    return batch_geodesic_distances.mean()


