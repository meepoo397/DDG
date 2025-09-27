from base.training.networks_edm2 import UNet, mp_sum,mp_cat,mp_silu
import numpy as np
import torch
from edm2impl.utils import debug_print
def ej_forward_constructor(debug=False,rho_noise=False,rho=1,z_normalize=True,theta=10,eps=1e-4):
    debug_print(debug,f'Using ej_forward with z_normalize={z_normalize},theta={theta}')
    def ej_forward(self, x, noise_labels, class_labels):
        # Embedding.
        emb = self.emb_noise(self.emb_fourier(noise_labels))
        if self.emb_label is not None:
            debug_print(debug,f'In ej_forward using theta noise')
            ##### The only place changed by EJ sampler ####
            # Original: emb = mp_sum(emb, self.emb_label(class_labels * np.sqrt(class_labels.shape[1])), t=self.label_balance)
            # EJ DDG disturb the embedding of label
            emb_label = self.emb_label(class_labels * np.sqrt(class_labels.shape[1]))
            if not rho_noise:
                z = emb_label # Shape (B,C)
                debug_print(debug,f'In ej_forward z.norm={z.norm(dim=1)}')
                r = torch.rand_like(z) # Shape (B,C)
                dot = (r * z).sum(dim=1, keepdim=True)  # Shape (B, 1)
                u = r - dot / (z.norm(dim=1, keepdim=True)**2) * z
                u = u / (u.norm(dim=1, keepdim=True) + eps) # Use eps to avoid resample u
                theta_torch = torch.tensor(theta * torch.pi / 180, device=z.device)
                z_prime = z + (torch.tan(theta_torch) * z.norm(dim=1, keepdim=True)) * u
                debug_print(debug,f'In ej_forward z_prime.norm={z_prime.norm(dim=1)}')
        
                if z_normalize:
                    z_prime_prime = z_prime  * (z.norm(dim=1, keepdim=True)/ z_prime.norm(dim=1, keepdim=True))
                else:
                    z_prime_prime = z_prime
                debug_print(debug,f'In ej_forward z_prime_prime.norm={z_prime_prime.norm(dim=1)}')
            else:
                debug_print(debug,f'In ej_forward using rho noise')
                z = emb_label # Shape (B,C)
                debug_print(debug,f'In ej_forward z.norm={z.norm(dim=1)}')
                r = torch.rand_like(z) # Shape (B,C)
                u = r / (r.norm(dim=1, keepdim=True) + eps) # Use eps to avoid resample u
                z_prime_prime = z + rho * u
                debug_print(debug,f'In ej_forward z_prime_prime.norm={z_prime_prime.norm(dim=1)}')
        
            # z_prime_prime = emb_label
            emb = mp_sum(emb, z_prime_prime, t=self.label_balance)

        emb = mp_silu(emb)

        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            x = block(x) if 'conv' in name else block(x, emb)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if 'block' in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)
        x = self.out_conv(x, gain=self.out_gain)
        return x
    return ej_forward
