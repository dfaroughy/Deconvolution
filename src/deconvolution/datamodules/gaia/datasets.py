import torch
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass

from deconvolution.datamodules.gaia.dataprocess import PreProcessGaiaData

class GaiaDataset(Dataset):

    def __init__(self, configs: dataclass):
        
        self.dataset = configs.DATASET
        self.cuts = configs.CUTS
        self.preprocess_methods = configs.PREPROCESS 
        self.summary_stats = None
        
        ''' data attributes:
            - target: gaia phase-space data (x, y, z, vx, vy, vz)
            - target_preprocessed:  gaia data with cuts and preprocessing
            - source: std gaussian noise
            - covariance: covariance matrix of gaia observations uncertainites
        '''

        self.get_covariance_data()
        self.get_target_data()
        self.get_source_data()

    def __getitem__(self, idx):
        output = {}
        output['target'] = self.target_preprocess[idx]
        output['source'] = self.source[idx]
        output['covariance'] = self.covariance[idx]
        output['mask'] = torch.ones_like(self.target_preprocess[idx][..., 0])
        output['context'] = torch.empty_like(self.target_preprocess[idx][..., 0])
        return output

    def __len__(self):
        return self.target_preprocess.size(0)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_target_data(self):
        target = torch.tensor(np.load(self.dataset[0]), dtype=torch.float32)        
        target = PreProcessGaiaData(target, cuts=self.cuts, methods=self.preprocess_methods)
        target.apply_cuts()
        self.target = target.features.clone()
        target.preprocess()
        self.summary_stats = target.summary_stats
        self.target_preprocess = target.features.clone()

    # def get_target_data(self):
    #     self.clean = torch.tensor(np.load(self.dataset[0]), dtype=torch.float32)    
    #     noise_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(6), self.covariance)
    #     self.noise = noise_dist.sample((self.clean.shape[0],))
    #     self.smeared = self.clean + self.noise
    #     target = PreProcessGaiaData(self.smeared, cuts=self.cuts, methods=self.preprocess_methods)
    #     target.apply_cuts()
    #     self.target = target.features.clone()
    #     target.preprocess()
    #     self.summary_stats = target.summary_stats
    #     self.target_preprocess = target.features.clone()

    def get_covariance_data(self):
        self.covariance = torch.tensor(np.load(self.dataset[1]), dtype=torch.float32)

    def get_source_data(self):
        self.source = torch.randn_like(self.target, dtype=torch.float32)
