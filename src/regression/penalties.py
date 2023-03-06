import torch
import torch.nn as nn

def diag_sum(vec: torch.Tensor) -> torch.Tensor:
    "q00 + q11 + q22 + q33 == 1"
    selected_entries = [0, 4, 7, 9]
    norm_sq = torch.sum(vec[:, selected_entries], dim=1)
    norm_penalty = torch.mean((norm_sq - 1)**2)
    return norm_penalty


def dot_01(vec: torch.Tensor) -> torch.Tensor:
    "q00 q11 == q01^2"
    dot_ = torch.sum(vec[:,0]*vec[:,4])
    sqr_ = torch.sum(vec[:,1]**2)
    penalty = torch.mean((dot_ - sqr_)**2)
    return penalty

def dot_02(vec: torch.Tensor) -> torch.Tensor:
    "q00 q22 == q02^2"
    dot_ = torch.sum(vec[:,0]*vec[:,7])
    sqr_ = torch.sum(vec[:,2]**2)
    penalty = torch.mean((dot_ - sqr_)**2)
    return penalty

def dot_03(vec: torch.Tensor) -> torch.Tensor:
    "q00 q33 == q03^2"
    dot_ = torch.sum(vec[:,0]*vec[:,9])
    sqr_ = torch.sum(vec[:,3]**2)
    penalty = torch.mean((dot_ - sqr_)**2)
    return penalty

def dot_12(vec: torch.Tensor) -> torch.Tensor:
    "q11 q22 == q12^2"
    dot_ = torch.sum(vec[:,4]*vec[:,7])
    sqr_ = torch.sum(vec[:,5]**2)
    penalty = torch.mean((dot_ - sqr_)**2)
    return penalty

def dot_13(vec: torch.Tensor) -> torch.Tensor:
    "q11 q33 == q13^2"
    dot_ = torch.sum(vec[:,4]*vec[:,9])
    sqr_ = torch.sum(vec[:,6]**2)
    penalty = torch.mean((dot_ - sqr_)**2)
    return penalty

def dot_23(vec: torch.Tensor) -> torch.Tensor:
    "q22 q33 == q23^2"
    dot_ = torch.sum(vec[:,7]*vec[:,9])
    sqr_ = torch.sum(vec[:,8]**2)
    penalty = torch.mean((dot_ - sqr_)**2)
    return penalty

def penalty_sum(vec: torch.Tensor) -> torch.Tensor:
    total = diag_sum(vec) + dot_01(vec) + dot_02(vec) + dot_03(vec) + dot_13(vec) + dot_23(vec) + dot_12(vec)

    return total

