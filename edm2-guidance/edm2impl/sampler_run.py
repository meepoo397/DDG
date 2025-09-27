import torch
import numpy as np
import tqdm
from typing import Callable,Optional

from base.training.encoders import Encoder
from utils import debug_print
from sampler.scheduler import guidance_scheduler_config

Network = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
Classifier = Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
class Sampler:
    def __init__(
            self, net: Network, gnet: Network, encoder: Encoder,
            classifier:Classifier,
            num_steps = 32, guidance_scheduler = 'const_scheduler', guidance = 1.7,
            dtype = torch.float32, sigma_min = 0.002, sigma_max = 80, rho = 7,
            S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, randn_like=torch.randn_like,
            pca = None, debug = False,
            **sampler_kwargs,
        ) -> None:
        self.net = net
        self.gnet = gnet
        self.encoder = encoder
        self.classifier = classifier
        self.num_steps = num_steps
        if guidance_scheduler not in guidance_scheduler_config:
            raise KeyError(f"Unknown guidance_scheduler: {guidance_scheduler}")
        self.guidance_scheduler = guidance_scheduler_config[guidance_scheduler]
        self.guidance = guidance
        self.dtype = dtype
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise
        self.randn_like = randn_like
        self.pca = pca
        self.debug = debug
        self.sampler_kwargs = sampler_kwargs
        self.heun_guid = sampler_kwargs.get('heun_guid', False)
        if 'heun_guid' not in sampler_kwargs:
            debug_print(self.debug,f"heun_guid is not specified using heun_guid=False")
        else:
            debug_print(self.debug,f"heun_guid={self.heun_guid}")
        
    
    def sample(self, noise: torch.Tensor, labels: torch.Tensor, seeds: list):
        # Compute seeds_array for any sampler that needs seed per step
        seeds_array = np.concatenate([
            np.random.default_rng(seed).integers(low=0, high=2**31, size=(self.num_steps,1))
            for seed in seeds], axis=1)
        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=self.dtype, device=noise.device)
        # In EDM sigma = t
        t_steps = (self.sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

        pca_features_in_process = []
        # Main sampling loop.
        x_next = noise.to(self.dtype) * t_steps[0]
        if self.pca is not None:
            flattened = x_next.detach().to("cpu").view(x_next.size(0), -1).numpy()
            pca_features_in_process.append(self.pca.transform(flattened))
        # for i, (t_cur, t_next) in tqdm.tqdm(enumerate(zip(t_steps[:-1], t_steps[1:])), total=len(t_steps)-1, desc=f'Denoising', ncols=100, unit='step'): # 0, ..., N-1
        for i, (t_cur, t_next,seed) in enumerate(zip(t_steps[:-1], t_steps[1:],seeds_array)): # 0, ..., N-1
            debug_print(self.debug, f"step {i}/{self.num_steps}")
            x_cur = x_next

            # Increase noise temporarily.
            if self.S_churn > 0 and self.S_min <= t_cur <= self.S_max:
                gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1)
                t_hat = t_cur + gamma * t_cur
                x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * self.randn_like(x_cur)
            else:
                t_hat = t_cur
                x_hat = x_cur
            if torch.isnan(x_hat).any():
                debug_print(self.debug,f"step {i}: x_hat is NaN")

            # Euler step.
            d_cur = (x_hat - self.denoise(x_hat, t_hat, labels, i,seed)) / t_hat
            if torch.isnan(d_cur).any():
                debug_print(self.debug,f"step {i}: d_cur is NaN")
            x_next = x_hat + (t_next - t_hat) * d_cur
            if torch.isnan(x_next).any():
                debug_print(self.debug,f"step {i}: x_next is NaN")

            # Apply 2nd order correction.
            debug_print(self.debug,f"step {i} Huen")
            if i < self.num_steps - 1:
                if self.heun_guid:
                    d_prime = (x_next - self.denoise(x_next, t_next, labels, i+1,seed)) / t_next
                else:
                    d_prime = (x_next - self.net(x_next, t_next, labels).to(self.dtype))/ t_next
                    
                if torch.isnan(d_prime).any():
                    debug_print(self.debug,f"step {i}: d_prime is NaN")
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                if torch.isnan(x_next).any():
                    debug_print(self.debug,f"step {i}: x_next (after correction) is NaN")
                
            # PCA features transforming
            if self.pca is not None:
                flattened = x_next.detach().to("cpu").view(x_next.size(0), -1).numpy()
                pca_features_in_process.append(self.pca.transform(flattened))
        
        if self.pca is not None:
            # There should be T=num_steps+1 trajactory
            assert len(pca_features_in_process)==self.num_steps + 1
            # For pca features
            # List[T of [B, C]] → [T, B, C]
            # Tranpose to [B, T, C]
            return x_next,np.transpose(np.stack(pca_features_in_process, axis=0), (1, 0, 2))
        else:
            return x_next,None
        
    def denoise(self, x: torch.Tensor, sigma: torch.Tensor, labels: torch.Tensor, i: int,seed: torch.Tensor):
        # sigma : noise scale sigma at time i
        # i     : iterate from 0,1,...,num_steps-1
        # num_steps: N=32 in EDM2
        Dx = self.net(x, sigma, labels).to(self.dtype)
        if not isinstance(Dx, torch.Tensor):
            debug_print(self.debug,f"Dx type is {type(Dx)}")
        if torch.isnan(Dx).any(): print("Dx is NaN!")
        # Compute scheduled guidance scale
        scheduled_guidance = self.guidance_scheduler(i, self.num_steps, self.guidance,t=sigma,**self.sampler_kwargs)
        debug_print(self.debug,f"Using original CFG sampler")
        debug_print(self.debug,f"scheduled_guidance: {scheduled_guidance}")
        
        if scheduled_guidance == 1:
            return Dx
        ref_Dx = self.gnet(x, sigma, labels).to(self.dtype)
        return ref_Dx.lerp(Dx, scheduled_guidance)

class SamplerWithClassifier(Sampler):
    def __init__(
            self,
            net: Network,
            gnet: Network,
            classifier: Classifier,   
            encoder: Encoder, num_steps=32, guidance_scheduler='const_scheduler', guidance=1.7, dtype=torch.float32, sigma_min=0.002, sigma_max=80,
            rho=7, S_churn=0, S_min=0, S_max=..., S_noise=1, randn_like=torch.randn_like, pca=None, debug=False,
            **sampler_kwargs,
        ) -> None:
        super().__init__(
            net, gnet, encoder,classifier, num_steps, guidance_scheduler, guidance, 
            dtype, sigma_min, sigma_max, rho, S_churn, S_min, S_max, S_noise, randn_like, pca, debug,
            **sampler_kwargs,
        )
        assert self.classifier is not None
        self.n = sampler_kwargs.get('n_random', 3)  # 若需要多個負樣本，可設 top_k > 1
        self.k = sampler_kwargs.get('k_average', 1)  # 平均n個score
        
    
    def classify(self, x: torch.Tensor, i: int) -> torch.Tensor:
        return self.classifier(x, torch.tensor([1000 * (1 - i / self.num_steps)] * x.shape[0], device=torch.device('cuda')))