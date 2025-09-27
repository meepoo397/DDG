import torch

from edm2impl.sampler_run import SamplerWithClassifier
from edm2impl.utils import debug_print

class MapNegDDGSampler(SamplerWithClassifier):
    def denoise(self, x: torch.Tensor, sigma: torch.Tensor, labels: torch.Tensor, i: int):
        label_idx = torch.argmax(labels, dim=1)
        
        # Not considering time embedding for simplicity
        logits = self.classify(self.encoder.decode(x).to(torch.float32) / 127.5 - 1, i)
        log_proba = torch.log_softmax(logits, dim=-1)
        log_proba[:, label_idx] = -torch.inf

        Dx = self.net(x, sigma, labels).to(self.dtype)
        if not isinstance(Dx, torch.Tensor):
            debug_print(self.debug, f"Dx type is {type(Dx)}")
        if torch.isnan(Dx).any(): print("Dx is NaN!")
        
        # Compute scheduled guidance scale
        scheduled_guidance = self.guidance_scheduler(i, self.num_steps, self.guidance)
        
        debug_print(self.debug, f"scheduled_guidance: {scheduled_guidance}")
        if scheduled_guidance == 1:
            return Dx
        ref_Dx = self.net(x, sigma, torch.eye(self.net.label_dim, device=torch.device('cuda'))[torch.argmax(log_proba, dim=1)]).to(self.dtype)
        return ref_Dx.lerp(Dx, scheduled_guidance)