import torch

from edm2impl.sampler_run import SamplerWithClassifier
from edm2impl.utils import debug_print
import random
import numpy as np

class CFGAdaptiveSamplerV13(SamplerWithClassifier):
    def __init__(
            self,
            net,
            gnet,
            classifier,   
            encoder, num_steps=32, guidance_scheduler='const_scheduler', guidance=1.7, dtype=torch.float32, sigma_min=0.002, sigma_max=80,
            rho=7, S_churn=0, S_min=0, S_max=..., S_noise=1, randn_like=torch.randn_like, pca=None, debug=False,
            **sampler_kwargs,
        ) -> None:
        super().__init__(
            net, gnet,classifier,encoder,num_steps=num_steps, guidance_scheduler=guidance_scheduler, guidance=guidance, 
            dtype=dtype, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, S_churn=S_churn, S_min=S_min, S_max=S_max, S_noise=S_noise, randn_like=randn_like, pca=pca, debug=debug,
            **sampler_kwargs,
        )
        self.w_interval_low_sigma = sampler_kwargs.get('w_interval_low_sigma', 0.28)
        self.w_interval_high_sigma = sampler_kwargs.get('w_interval_high_sigma', 2.9)
        self.min_guidance = sampler_kwargs.get('min_guidance', 1.75)
        self.max_guidance = sampler_kwargs.get('max_guidance', 2.8)
        self.open_prob = sampler_kwargs.get('open_prob', 0.5)
        self.balance_coeff = sampler_kwargs.get('balance_coeff', 3)
        self.curve_data = sampler_kwargs.get('curve_data', 3)
        debug_print(self.debug,f'Using CFGAdaptiveSampler with min_guidance:{self.min_guidance},max_guidance:{self.max_guidance},w_interval_low_sigma:{self.w_interval_low_sigma},w_interval_high_sigma:{self.w_interval_high_sigma}')
    

    def shift_and_scale_curve(self,data: list, min_guidance: float, max_guidance: float,
                          num_steps: int,  t_steps,
                          interval_sigma_min: float = 0.28, interval_sigma_max: float = 2.9,
                          set_to_value: float = 1.0) -> np.ndarray:
        if not data:
            return np.array([])
        
        data = np.array(data)
        # 為了與你提供的數據長度匹配，我們只考慮前 num_steps 個 t_steps
        if len(t_steps) > len(data):
            t_steps = t_steps[:len(data)]
            t_steps = t_steps.to('cpu')
        
        is_guided = (t_steps >= interval_sigma_min) & (t_steps <= interval_sigma_max)
        
        guided_data = data[is_guided]
        
        new_data = np.full_like(data, set_to_value, dtype=float)
    
        if guided_data.size > 0:
            min_val = np.min(guided_data)
            max_val = np.max(guided_data)
            
            # 避免除以零
            if not np.isclose(min_val, max_val):
                # 對符合條件的數據進行 Rescale
                scaled_guided_data = (guided_data - min_val) / (max_val - min_val)
                transformed_guided_data = scaled_guided_data * (max_guidance - min_guidance) + min_guidance
                
                # 將轉換後的數據放回新的陣列中
                new_data[is_guided] = transformed_guided_data
            else:
                # 如果符合條件的數據範圍為零，則全部設定為 min_guidance
                new_data[is_guided] = min_guidance
        
        return new_data
    
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

        if self.curve_data == 'edm2_xxl_cfg_const_10000':
            # print('using edm2_xxl_cfg_const_10000')
            original_data = [0, -1.991400495171547e-05, 2.223765477538109e-06, 9.581749327480793e-05, 0.00020839041098952293, 0.00016641197726130486, 0.00016891141422092915, 0.00023579876869916916, 0.0003273698966950178, 0.0005652119871228933, 0.0012883222661912441, 0.002721992786973715, 0.006421000696718693, 0.014279724098742008, 0.030967721715569496, 0.05095775052905083, 0.07450365275144577, 0.09355135262012482, 0.10675257444381714, 0.10174217820167542, 0.07791346311569214, 0.04631483554840088, 0.019972264766693115, 0.004866600036621094, -0.0015451312065124512, -0.003330707550048828, -0.0028957724571228027, -0.0018478631973266602, -0.0009563565254211426, -0.00038743019104003906, -4.684925079345703e-05, 0.0009469985961914062]
        else:# CFG Interval edm2 s 
            original_data = [0, 0.00011854968033730984, -6.204703822731972e-06, 7.344386540353298e-05, 8.147815242409706e-05, 7.347413338720798e-05, 0.00013053789734840393, 0.00024396670050919056, 0.00034406152553856373, 0.0004322801250964403, 0.0007102922536432743, 0.0017681904137134552, 0.00401817774400115, 0.00889554899185896, 0.019199594855308533, 0.033939484506845474, 0.06823297590017319, 0.1106177419424057, 0.12362787127494812, 0.11659982800483704, 0.09012675285339355, 0.053175389766693115, 0.02519327402114868, 0.008260965347290039, 0.0005714893341064453, -0.0017622709274291992, -0.0018764138221740723, -0.0013113021850585938, -0.0007520318031311035, -0.00029480457305908203, -0.0004126429557800293, -7.152557373046875e-07]
        
        w = [(1 + self.balance_coeff * diff_prob) if diff_prob > self.open_prob else 1 for diff_prob in original_data]
        # adaptive_scheduler_state = torch.zeros((noise.shape[0],2),dtype=torch.int32)
        
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
            
            
            # Calculate candidate by self.classifier
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

            # Compute w_t
            w_t = w[i]

            
            # Euler step.
            d_cur = (x_hat - self.denoise_with_candidate(x_hat, t_hat, labels, i,w_t)) / t_hat
            if torch.isnan(d_cur).any():
                debug_print(self.debug,f"step {i}: d_cur is NaN")
            x_next = x_hat + (t_next - t_hat) * d_cur
            if torch.isnan(x_next).any():
                debug_print(self.debug,f"step {i}: x_next is NaN")

            # Apply 2nd order correction.
            debug_print(self.debug,f"step {i} Huen")
            if i < self.num_steps - 1:
                if self.heun_guid:
                    d_prime = (x_next - self.denoise_with_candidate(x_next, t_next, labels, i+1,w_t)) / t_next
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
    def denoise_with_candidate(self, x: torch.Tensor, sigma: torch.Tensor, labels: torch.Tensor, i: int,w_t:float):
        # t : noise scale sigma at time i
        # i : iterate from 0,1,...,num_steps-1
        # num_steps: N in EDM2
        debug_print(self.debug,f"In denoise_with_candidate sigma:{sigma}")
        Dx = self.net(x, sigma, labels).to(self.dtype)
        if not isinstance(Dx, torch.Tensor):
            debug_print(self.debug,f"Dx type is {type(Dx)}")
        if torch.isnan(Dx).any(): print("Dx is NaN!")
        # Compute scheduled guidance scale
        
        ref_Dx = self.gnet(x, sigma, labels).to(self.dtype)
        # print(f'Dx.shape:{Dx.shape}')
        # print(f'step:{i},w_t:{w_t}')
        return ref_Dx.lerp(Dx, w_t)