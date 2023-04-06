import torch
import torch.nn as nn
import collections
from typing import Dict, Callable, List

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

def constrain_dict() -> Dict[int, Callable[[torch.Tensor], torch.Tensor]]:
    return {1:diag_sum, 2:dot_01, 3:dot_02, 4:dot_03,
            5:dot_12, 6:dot_13, 7:dot_23}

def penalty_sum(vec: torch.Tensor, apply_constrain: List[int]) -> torch.Tensor:
    total = 0
    dict_ = constrain_dict()
    for idx in apply_constrain:
        total += dict_[idx](vec)

    return total

