import torch
import tqdm
import numpy as np

from edm2impl.sampler_run import SamplerWithClassifier
from edm2impl.utils import debug_print

class CGSampler(SamplerWithClassifier):
    def sample(self, noise: torch.Tensor, labels: torch.Tensor, seeds: list):
        step_indices = torch.arange(self.num_steps, dtype=self.dtype, device=noise.device)
        # In EDM sigma = t
        t_steps = (self.sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

        pca_features_in_process = []
        # Main sampling loop.
        x = noise.to(self.dtype) * t_steps[0]
        if self.pca is not None:
            flattened = x.detach().to("cpu").view(x.size(0), -1).numpy()
            pca_features_in_process.append(self.pca.transform(flattened))
        for i, t_cur in tqdm.tqdm(enumerate(t_steps[:-1]), total=len(t_steps)-1, desc=f'Denoising', ncols=100, unit='step'): # 0, ..., N-1
            debug_print(self.debug, f"step {i}/{self.num_steps}")
            
            label_idx = torch.argmax(labels, dim=1)
            
            x_0 = self.encoder.decode(self.net(x, t_cur, labels)).to(self.dtype) / 127.5 - 1
            x_decoded = (self.encoder.decode(x).to(self.dtype) / 127.5 - 1).detach().requires_grad_(True)
            
            if torch.isnan(x_0).any():
                debug_print(self.debug, f"step {i}: x_0 is NaN")
            if torch.isnan(x_decoded).any():
                debug_print(self.debug, f"step {i}: x_decoded is NaN")

            logits = self.classify(x_decoded, i)
            logit_proba = torch.log_softmax(logits, dim=-1)
            if torch.isnan(logit_proba).any():
                debug_print(self.debug, f"step {i}: logit_proba is NaN")

            grad = torch.autograd.grad(logit_proba[range(x_decoded.shape[0]), label_idx].sum(), x_decoded)[0]
            if torch.isnan(grad).any():
                debug_print(self.debug, f"step {i}: grad is NaN")
            
            x_0 += self.guidance_scheduler(i, self.num_steps, self.guidance,t) * t_cur.sum() ** 2 * grad
            x_0 = ((x_0 + 1) * 127.5).to(torch.int32)
            x = self.randn_like(x) * t_cur + self.encoder.encode(x_0)
            if torch.isnan(grad).any():
                debug_print(self.debug, f"step {i}: x (after denoising) is NaN")
            
            # PCA features transforming
            if self.pca is not None:
                flattened = x.detach().to("cpu").view(x.size(0), -1).numpy()
                pca_features_in_process.append(self.pca.transform(flattened))
        
        if self.pca is not None:
            # There should be T=num_steps+1 trajactory
            assert len(pca_features_in_process)==self.num_steps + 1
            # For pca features
            # List[T of [B, C]] → [T, B, C]
            # Tranpose to [B, T, C]
            return x, np.transpose(np.stack(pca_features_in_process, axis=0), (1, 0, 2))
        else:
            return x, None