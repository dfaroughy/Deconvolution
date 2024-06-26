from typing import List
from dataclasses import dataclass, field

""" Default configurations for training.
"""

@dataclass
class Optimizer_Configs:
    DEVICE : str = 'cpu'
    OPTIMIZER: str = 'Adam'
    LR : float = 1e-4
    WEIGHT_DECAY : float = 0.0
    OPTIMIZER_BETAS : List[float] = field(default_factory = lambda : [0.9, 0.999])
    OPTIMIZER_EPS : float = 1e-8
    OPTIMIZER_AMSGRAD : bool = False
    GRADIENT_CLIP : float = None

@dataclass
class Scheduler_Configs:
    SCHEDULER: str = None
    SCHEDULER_T_MAX: int = None
    SCHEDULER_ETA_MIN: float = None
    SCHEDULER_GAMMA: float = None
    SCHEDULER_STEP_SIZE: int = None

@dataclass
class Training_Configs(Scheduler_Configs, Optimizer_Configs):
    EPOCHS: int = 10
    BATCH_SIZE : int = 256
    DATA_SPLIT_FRACS : List[float] = field(default_factory = lambda : [1.0, 0.0, 0.0])  # train / val / test 
    NUM_WORKERS : int = 0
    PIN_MEMORY: bool = False
    EARLY_STOPPING : int = None
    MIN_EPOCHS : int = None 
    PRINT_EPOCHS : int = None   
    FIX_SEED : int = None