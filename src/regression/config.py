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
    num_hidden: int = 2

@dataclasses.dataclass
class FeedForwardTrainConfig:
    data: TrainingDataConfig = TrainingDataConfig()
    model_config: NetworkConfig = NetworkConfig()
    optim: OptimConfig = OptimConfig()
    batch_size: int = 64
    num_epochs: int = 10
    device: str = 'cpu'
    num_gpus: int = 1
    log_every: int = 1

@dataclasses.dataclass
class TestConfig(FeedForwardTrainConfig):
    chkpt_path: str = omegaconf.MISSING 