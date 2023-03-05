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
    #option for model net
    category: List[str] = dataclasses.field(default_factory=lambda:["airplane"])
    sigma: float = 0.01
    num_points: int = 8000
    model_net: bool = False


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
    num_out: int = 4
    hidden_size: int = 1024
    adj_option: str = 'adjugate' #by default, also can be 'a-matrix/chordal'
    batch_norm: bool = False

@dataclasses.dataclass
class PointNetTrainConfig:
    data: TrainingDataConfig = TrainingDataConfig()
    model_config: NetworkConfig = NetworkConfig()
    optim: OptimConfig = OptimConfig()
    batch_size: int = 64
    num_epochs: int = 10
    device: str = 'gpu'
    num_gpus: int = 1
    log_every: int = 1
    constrain: bool = False

@dataclasses.dataclass
class TestConfig(PointNetTrainConfig):
    chkpt_path: str = omegaconf.MISSING 