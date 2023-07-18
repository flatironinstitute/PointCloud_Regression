import dataclasses
from typing import Optional
import omegaconf

@dataclasses.dataclass
class SimulatorConfig:
    """Config for generating training data
    Paras
    ------
    device: 
        the torch device to use gor generating
    """
    device: str = 'cpu'
    rotation_format: str = 'zxy' 
    output_path: str = omegaconf.MISSING
    max_angle: int = 180
    batch_size: int = 100
    num_points: int = 100
    sigma: float = 0.01
    source_norm: bool = False
    one_source: bool = False
    uniform_: bool = False #random distribution to generate source cloud, Gaussian by default
    manual: bool = False #whether use our manually random quaternion or not

