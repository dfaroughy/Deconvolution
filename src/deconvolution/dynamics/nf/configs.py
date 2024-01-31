from dataclasses import dataclass

""" Default configurations for discrete normalizing flow dynamics.
"""

@dataclass
class NormFlow_Configs:
    DYNAMICS: str = 'NormFlow'
    PERMUTATION: str = '1-cycle'
    NUM_TRANSFORMS: int = 5
    
@dataclass
class Deconvolution_Configs:
    DYNAMICS: str = 'DeconvolutionFlow'
    PERMUTATION: str = '1-cycle'
    NUM_TRANSFORMS: int = 5
    NUM_NOISE_DRAWS: int = 30
