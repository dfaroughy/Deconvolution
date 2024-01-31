from dataclasses import dataclass
from deconvolution.trainer.configs import Training_Configs

""" Default configurations for models.
"""

@dataclass
class MLP_Configs(Training_Configs):
    MODEL : str = 'MLP'
    DIM_HIDDEN : int = 128   
    DIM_TIME_EMB : int = None
    NUM_LAYERS : int = 3
    ACTIVATION : str = 'ReLU'

#...Normalizing Flow Models:

@dataclass
class MAF_Affine_Configs(Training_Configs):
    MODEL : str = 'MAF_Affine'
    DIM_HIDDEN : int = 128 
    NUM_BLOCKS: int = 2 
    USE_RESIDUAL_BLOCKS: bool = False
    DROPOUT : float = 0.0
    USE_BATCH_NORM : bool = False

@dataclass
class Couplings_Affine_Configs(Training_Configs):
    MODEL : str = 'Couplings_Affine'
    MASK : str = 'checkerboard'
    DIM_HIDDEN : int = 128 
    NUM_BLOCKS: int = 2 
    USE_RESIDUAL_BLOCKS: bool = False
    DROPOUT : float = 0.0
    USE_BATCH_NORM : bool = False

@dataclass
class MAF_RQS_Configs(Training_Configs):
    MODEL : str = 'MAF_RQS'
    DIM_HIDDEN : int = 128 
    NUM_BLOCKS: int = 2 
    USE_RESIDUAL_BLOCKS: bool= False
    DROPOUT : float = 0.0
    USE_BATCH_NORM : bool = False
    NUM_BINS : int = 10
    TAILS : str = 'linear'
    TAIL_BOUND : int = 5

@dataclass
class Couplings_RQS_Configs(Training_Configs):
    MODEL : str = 'Couplings_RQS'
    MASK : str = 'checkerboard'
    DIM_HIDDEN : int = 128 
    NUM_BLOCKS: int = 2 
    USE_RESIDUAL_BLOCKS: bool= False
    DROPOUT : float = 0.0
    USE_BATCH_NORM : bool = False
    NUM_BINS : int = 10
    TAILS : str = 'linear'
    TAIL_BOUND : int = 5