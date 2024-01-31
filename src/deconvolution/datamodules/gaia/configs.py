import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

""" Default configurations for Gaia datasets.
"""

@dataclass
class Gaia_Configs:
    DATA : str = 'Gaia'
    DATASET : List[str] = field(default_factory = lambda : ['../../data/gaia/data.angle_340.smeared_00.npy',
                                                            '../../data/gaia/data.angle_340.smeared_00.cov.npy'])
    FEATURES : List[str] = field(default_factory = lambda : ['x', 'y', 'z', 'vx', 'vy', 'vz'])
    DIM_INPUT : int = 6
    R_SUN: List[float] = field(default_factory = lambda :[8.122, 0.0, 0.0208])
    PREPROCESS : List[str] = field(default_factory = lambda : ['unit_ball_transform', 'standardize' ])
    CUTS : Dict[str, List[float]] = field(default_factory = lambda: {'r': [0.0, 3.5]} )
    
@dataclass
class Deconvolve_Gaia_Configs:
    DATA : str = 'DeconvolveGaia'
    DATASET : List[str] = field(default_factory = lambda : ['../../data/gaia/data.angle_340.smeared_00.npy',
                                                            '../../data/gaia/data.angle_340.smeared_00.cov.npy'])
    FEATURES : List[str] = field(default_factory = lambda : ['x', 'y', 'z', 'vx', 'vy', 'vz'])
    DIM_INPUT : int = 6
    R_SUN: List[float] = field(default_factory = lambda :[8.122, 0.0, 0.0208])
    PREPROCESS : List[str] = field(default_factory = lambda : ['unit_ball_transform', 'standardize' ])
    CUTS : Dict[str, List[float]] = field(default_factory = lambda: {'r': [0.0, 3.5]} )
    
