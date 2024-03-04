import torch
from dataclasses import dataclass

class NormalizingFlow:

    def __init__(self, configs: dataclass):
        self.dim = configs.DIM_INPUT
        self.device = configs.DEVICE
        self.num_transforms = configs.NUM_TRANSFORMS

    def loss(self, model, batch):
        ''' Negative Log-probability loss
        '''
        target = batch['target'].to(self.device)
        loss = - model.log_prob(target)
        return torch.mean(loss)

class DeconvolutionFlow:

    def __init__(self, configs: dataclass):
        self.dim = configs.DIM_INPUT
        self.device = configs.DEVICE
        self.num_transforms = configs.NUM_TRANSFORMS
        self.num_noise_draws = configs.NUM_NOISE_DRAWS
        self.stats = configs.DATA_STATS
        
    def loss(self, model, batch):
        """ deconvolution loss
        """
        cov = batch['covariance'] 
        smeared = batch['target'] 
        x_mu = self.stats['mean']
        x_std = self.stats['std']
        cov = cov.repeat_interleave(self.num_noise_draws,0)            # ABC... -> AABBCC...
        smeared = smeared.repeat_interleave(self.num_noise_draws,0)    # ABC... -> AABBCC...
        epsilon = torch.randn_like(smeared)
        epsilon = torch.reshape(epsilon,(-1, epsilon.dim(), 1)) 
        x = smeared - torch.squeeze(torch.bmm(cov, epsilon))        # x = smeared - cov * epsilon
        x = x.to(self.device)
        logprob = torch.reshape(model.log_prob((x - x_mu) / x_std) + torch.log(x_std), (-1, self.num_noise_draws)) 
        loss = - torch.mean(torch.logsumexp(logprob, dim=-1))
        return loss + torch.log(torch.tensor(self.num_noise_draws))




