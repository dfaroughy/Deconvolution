import torch
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass
from deconvolution.datamodules.toys.dataprocess import PreProcessGaussData

class SmearedGaussDataset(Dataset):

    ''' Creates 3 perpendicular gaussians in the 2D plane, 
        applies gaussian smearing with a given noise covariance. 

        output items:
            - smeared: 3 gaussians
            - covariance: noise covariance
    '''
        
    def __init__(self, configs: dataclass):
        
        self.num_points = configs.NUM_POINTS
        self.loc = configs.DATA_LOC
        self.scale = configs.DATA_SCALE
        self.noise_cov = torch.Tensor(configs.NOISE_COV)
        self.cuts = configs.CUTS
        self.preprocess_methods = configs.PREPROCESS 

        self.get_truth_data()
        self.get_target_data()
        self.get_cov_data()

    def __getitem__(self, idx):
        output = {}
        output['target'] = self.target_preprocess[idx] if self.preprocess_methods is not None else self.target[idx]
        output['covariance'] = self.covs[idx]
        output['mask'] = torch.ones_like(self.target[idx][..., 0])
        output['context'] = torch.empty_like(self.target[idx][..., 0])
        return output

    def __len__(self):
        return self.target.size(0)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_target_data(self):
        noise_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), self.noise_cov)
        noise = noise_dist.sample((self.truth.shape[0],))
        smeared = self.truth + noise
        if self.preprocess_methods is not None:
            target = PreProcessGaussData(smeared, cuts=self.cuts, methods=self.preprocess_methods)
            self.target = target.features.clone()
            target.preprocess()
            self.summary_stats = target.summary_stats
            self.target_preprocess = target.features.clone()
        else:
            self.target = smeared

    def get_cov_data(self):
        if self.preprocess_methods is not None:
            covs = self.noise_cov * self.summary_stats['std']
            self.covs = covs.unsqueeze(0).repeat(self.num_points, 1, 1)
        else:
            self.covs = self.noise_cov.unsqueeze(0).repeat(self.num_points, 1, 1)

    def get_truth_data(self):
            data_means = torch.Tensor([
                [-2.0, 0.0],
                [0.0, -2.0],
                [0.0,  2.0]
            ])
            data_covars = torch.Tensor([
                [[0.3**2, 0],[0, 1]],
                [[1, 0],[0, 0.3**2]],
                [[1, 0],[0, 0.3**2]]])
            distributions = [torch.distributions.multivariate_normal.MultivariateNormal(mean, covar) for mean, covar in zip(data_means, data_covars)]
            multi = torch.multinomial(torch.ones(len(distributions)), self.num_points, replacement=True)
            data = []
            for i in range(self.num_points):
                selected_distribution = distributions[multi[i]]
                sample = selected_distribution.sample()
                data.append(sample)
            self.truth = torch.stack(data) * self.scale + self.loc
