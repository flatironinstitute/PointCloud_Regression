import dataclasses
from typing import Optional, Tuple, List
import omegaconf

@dataclasses.dataclass
class TrainingDataConfig:
    """configuration of data loading

    attr:
    file_path(str): path of simulated point cloud and quaternion
    train_prop(float): percentage for training
    """
    file_path: str = omegaconf.MISSING 
    train_prop: float = 0.9
    limit: Optional[int] = None
    num_data_workers: int = 16
    svd_mod: bool = False
    #options for model net
    category: List[str] = dataclasses.field(default_factory=lambda:["airplane"])
    sigma: float = 0.01
    num_points: int = 1000 #downsampled size for the modelnet mesh
    num_rot: int = 1000
    model_net: bool = False
    range_max: int = 35000
    range_min: int = 30000


@dataclasses.dataclass
class OptimConfig:
    """hyperparams of optimization

    attr:
    learning_rate(float): lr of training
    """
    learning_rate: float = 1e-3

@dataclasses.dataclass
class NetworkConfig:
    num_points: int = 100
    num_layer: int = 3
    hidden_size: int = 1024
    adj_option: str = 'adjugate' #by default, also can be 'a-matrix/chordal/six-d/rmsd'
    batch_norm: bool = False

@dataclasses.dataclass
class LossConfig:
    rmsd_trace: bool = False

@dataclasses.dataclass
class PointNetTrainConfig:
    data: TrainingDataConfig = TrainingDataConfig()
    model_config: NetworkConfig = NetworkConfig()
    loss_config: LossConfig = LossConfig()
    optim: OptimConfig = OptimConfig()
    batch_size: int = 64
    num_epochs: int = 10
    device: str = 'gpu'
    num_gpus: int = 1
    log_every: int = 1
    constrain: bool = False
    cnstr_pre: float = 1.0
    select_constrain: List[int] = dataclasses.field(default_factory=lambda:[1])

@dataclasses.dataclass
class TestConfig(PointNetTrainConfig):
    chkpt_path: str = omegaconf.MISSING 