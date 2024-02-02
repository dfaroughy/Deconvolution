import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


""" Default configurations for deconvolution datasets.
"""

@dataclass
class Deconvolution_Gauss_Configs:
    DATA: str = 'Gauss2D'
    FEATURES : List[str] = field(default_factory = lambda : ['x', 'y'])
    NUM_POINTS : int = 10000
    DIM_INPUT : int = 2
    DATA_LOC : float = 0.0
    DATA_SCALE : float = 1.0
    NOISE_COV : List[List[float]] = field(default_factory = lambda : [[0.1, 0.0],[0.0, 1.0]])
    PREPROCESS : List[str] = field(default_factory = lambda : [])
    CUTS : Dict[str, List[float]] = field(default_factory = lambda: {'x': [-np.inf, np.inf], 'y': [-np.inf, np.inf]} )
    
