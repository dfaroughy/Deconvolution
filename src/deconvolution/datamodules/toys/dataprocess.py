from typing import Any
import torch
import numpy as np

class PreProcessGaussData:

    def __init__(self, 
                 data, 
                 cuts: dict=None,
                 summary_stats: dict=None,
                 methods: list=None,
                 ):
        
        self.features = data
        self.cuts = cuts if cuts is not None else {'x': None, 'y': None}
        self.methods = methods 
        self.summary_stats = {} if summary_stats is None else summary_stats

    def apply_cuts(self):
        self.selection_cuts(feature='x', cut=self.cuts['x'])
        self.selection_cuts(feature='y', cut=self.cuts['y'])

    def preprocess(self):        
        #...preprocess with provided methods
        for method in self.methods:
            method = getattr(self, method, None)
            if method and callable(method): method()
            else: raise ValueError('Preprocessing method {} not implemented'.format(method))
    
    def selection_cuts(self, feature, cut=None):
        if cut is None: cut=[-np.inf, np.inf]
        dic={'x':0, 'y':1}
        mask = (self.features[..., dic[feature]] >= cut[0]) & (self.features[..., dic[feature]] <= cut[1])
        self.features = self.features[mask]

    def center(self):
        """ center data to have zero mean
        """
        self.summary_stats['mean'] = torch.mean(self.features, dim=0)
        self.summary_stats['std'] = torch.ones_like(self.summary_stats['mean'])
        self.features = self.features - self.summary_stats['mean']

    def standardize(self,  sigma: float=1.0):
        """ standardize data to have zero mean and unit variance
        """
        self.summary_stats['mean'] = torch.mean(self.features, dim=0)
        self.summary_stats['std'] = torch.std(self.features, dim=0)
        self.features = (self.features - self.summary_stats['mean']) * (sigma / self.summary_stats['std'])


class PostProcessGaussData:

    def __init__(self, 
                 data, 
                 summary_stats,
                 methods: list=None
                 ):
        
        self.features = data
        self.summary_stats = summary_stats
        self.methods = methods

    def postprocess(self):
        for method in self.methods:
            method = getattr(self, method, None)
            if method and callable(method): method()
            else: raise ValueError('Postprocessing method {} not implemented'.format(method))

    def inverse_center(self):
        mean = self.summary_stats['mean'].to(self.features.device)
        self.features = self.features + mean

    def inverse_standardize(self, sigma: float=1.0):
        std = self.summary_stats['std'].to(self.features.device)
        mean = self.summary_stats['mean'].to(self.features.device)
        self.features = self.features * (std / sigma) + mean


