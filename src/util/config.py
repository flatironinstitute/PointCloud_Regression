import dataclasses
from typing import Optional, Tuple, List, Type
import omegaconf

@dataclasses.dataclass
class PascalInferConfig:
    file_path: str = omegaconf.MISSING 
    chk_path: str = omegaconf.MISSING
    output_path: str = omegaconf.MISSING 
    crop: int = 224
    option: str = "adjugate" # by default
    num_sample: int = 5000
    batch_size: int = 32
    device: str = 'gpu'
    num_gpus: int = 1
    log_every: int = 1
    # options for pascal3D+
    category: List[str] = dataclasses.field(default_factory=lambda:["aeroplane"])

@dataclasses.dataclass
class PascalFileCounterConfig:
    pascal_path: str = omegaconf.MISSING
    syn_path: str = omegaconf.MISSING
    category: List[str] = dataclasses.field(default_factory=lambda:["aeroplane"])