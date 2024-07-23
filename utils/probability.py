"""
Utility functions related to probabilities and statistics, e.g. log likelihoods, ...
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


__log_2_pi = np.log(2*np.pi)


class GaussianKernelConv(nn.Module):
    def __init__(self, k: int = 5, sigma: float = 0.02):
        super().__init__()
        self.k = k
        self.sigma = sigma
        self.cache = dict()

    def kernel(self, n):
        if n not in self.cache:
            self.cache[n] = torch.exp(-(torch.arange(-self.k, self.k + 1) / n / self.sigma) ** 2 / 2)
            self.cache[n] /= self.cache[n].sum()

        return self.cache[n]
    
    def forward(self, target):
        kernel = self.kernel(target.shape[-1]).unsqueeze(0).unsqueeze(1).type_as(target)
        
        with torch.no_grad():
            weights = F.conv1d(target.unsqueeze(1), kernel, padding='same')
            weights = F.normalize(weights.squeeze(1), p=1, dim=-1)
        
        return weights
    

def standard_gaussian_log_probability(samples):
    """
    Computes the log-probabilities of given batch of samples using a multivariate gaussian distribution
    of independent components (zero-mean, identity covariance matrix).
    """
    return -0.5 * (samples.shape[1] * __log_2_pi + torch.sum(samples**2, dim=1))


def gaussian_log_probability(samples, mu, log_var):
    """
    Computes the log-probabilities of given batch of samples using a multivariate gaussian distribution
    of independent components (diagonal covariance matrix).
    """
    # if samples and mu do not have the same size,
    # torch automatically properly performs the subtract if mu is 1 dim smaller than samples
    return -0.5 * (samples.shape[1] * __log_2_pi +
                   torch.sum( log_var + ((samples - mu)**2 / torch.exp(log_var)), dim=1))



