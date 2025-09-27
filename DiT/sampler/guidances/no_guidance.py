import torch
from edm2impl.sampler_run import Sampler

class NoGuidSampler(Sampler):
    def denoise(self, x: torch.Tensor, sigma: torch.Tensor, labels: torch.Tensor, i: int,seed: torch.Tensor):
        return self.net(x, sigma, labels).to(self.dtype)