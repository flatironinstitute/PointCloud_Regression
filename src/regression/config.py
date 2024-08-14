import dataclasses
from typing import Optional, Tuple, List, Type, Dict
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
    svd_mod: bool = False # use qrmsd or qinit as the g.t.
    # options for model net
    category: List[str] = dataclasses.field(default_factory=lambda:["airplane"])
    sigma: float = 0.01
    num_points: int = 1000 # downsampled size for the modelnet mesh
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
    adj_option: str = 'adjugate' # by default, also can be 'a-matrix/chordal/six-d/svd/rmsd'
    batch_norm: bool = False

@dataclasses.dataclass
class LossConfig:
    # special option to make loss as the euclidean dist between two cloud
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
class PascalDataConfig:
    """configuration of data loading for Pascal3D+ dataset

    attr:
    file_path(str): path of the Pascal3D's images and annotations
    train_prop(float): percentage for training
    """
    pascal_path: str = omegaconf.MISSING 
    imagenet_path: str = omegaconf.MISSING
    syn_path:str = omegaconf.MISSING
    num_sample: int = 400000
    crop: int = 224
    train_prop: float = 0.9
    limit: Optional[int] = None
    num_data_workers: int = 8
    # options for pascal3D+
    category: List[str] = dataclasses.field(default_factory=lambda:["aeroplane"])
    sample_weights: Dict[str,float] = dataclasses.field(default_factory=lambda: 
    {
        'pascal': 1.0,
        'imagenet': 0.5,
        'synthetic': 0.2
    })
    
@dataclasses.dataclass
class RegNetConfig:
    n_class:int = 12 # by default, should be updated if we only train/eval several classes
    regress_option: str = 'adjugate' # by default, also can be 'a-matrix/chordal/svd'
    batch_norm: bool = False

@dataclasses.dataclass
class RegNetTrainingConfig:
    data: PascalDataConfig = PascalDataConfig()
    network: RegNetConfig = RegNetConfig()
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