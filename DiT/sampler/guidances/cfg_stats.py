import torch

from edm2impl.sampler_run import SamplerWithClassifier
from edm2impl.utils import debug_print
import random
import numpy as np
import torchvision.transforms as transforms

class CFGStatsSampler(SamplerWithClassifier):
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
        self.avoid_self_label = sampler_kwargs.get('avoid_self_label', True)  # 若需要多個負樣本，可設 top_k > 1
        debug_print(self.debug,f'Using MapNegDDGSampler_x0pred with avoid_self_label:{self.avoid_self_label}')
        self.k = sampler_kwargs.get('k_average', 1)  # 平均n個score
    def calculate_candidate(self,x:torch.Tensor,sigma:torch.Tensor,true_labels:torch.Tensor,i):
        debug_print(self.debug,f'true_labels.shape:{true_labels.shape}')
        # debug_print(self.debug,f'true_labels:{true_labels}')
        debug_print(self.debug,f'sigma.shape:{sigma.shape}')
        debug_print(self.debug,f'sigma:{sigma}')
        sigma = sigma.expand(x.size(0))  # 複製成 batch size 長度的 tensor
        debug_print(self.debug,f'After reshape sigma.shape:{sigma.shape}')
        # debug_print(self.debug,f'After reshape sigma:{sigma}')
        ## 1. Original
        # logits = self.classifier(x,sigma)
        ## 2. x0 prediction with x,t input
        xo_predicted = self.net(x,sigma,true_labels)
        dummy_t = torch.ones(true_labels.shape[0]).to(true_labels.device)
        ###以下為Imagenet64 res_50_9所用
        xo_predicted_cpu = (xo_predicted.cpu() * 0.5 + 0.5).clamp(0, 1)
        images_224 = torch.stack([self.preprocess(transforms.ToPILImage()(img)) for img in xo_predicted_cpu])
        images_224 = images_224.to(true_labels.device)
        xo_predicted = images_224
        logits = self.classifier(xo_predicted)
        ##以上為Imagenet64 res_50_9所用
        ##以下為其餘
        # logits = self.classifier(xo_predicted,dummy_t)
        #以上為其餘
        ## 3. x0 prediction with x input
        # logits = self.classifier(x)
        log_proba = torch.log_softmax(logits, dim=-1)
        true_labels_flat = true_labels.argmax(dim=-1).long()
        debug_print(self.debug,f'true_labels_flat.shape:{true_labels_flat.shape}')
        debug_print(self.debug,f'true_labels_flat:{true_labels_flat}')
        # debug_print(self.debug,f'log_proba before -inf self index:{log_proba}')
        _, top_ks_for_check_correct = torch.topk(log_proba, self.n, dim=-1)
        correct = (top_ks_for_check_correct == true_labels_flat[:, None]).float().sum(dim=-1).mean().item()
        debug_print(self.debug,f'correct:{correct}')
        debug_print(self.debug,f'correct:{correct}')
        if self.avoid_self_label:
            log_proba[torch.arange(logits.size(0)), true_labels_flat] = -torch.inf
        # debug_print(self.debug,f'log_proba after -inf self index:{log_proba}')
        _, top_ks = torch.topk(log_proba, self.n, dim=-1)
        debug_print(self.debug,f'top_ks.shape:{top_ks.shape}')
        debug_print(self.debug,f'calculate_candidate get:{top_ks}')
        
        probabilities = torch.softmax(logits, dim=-1)

        # 處理真實標籤 (true_labels)
        if true_labels.dim() > 1:
            true_labels_flat = true_labels.argmax(dim=-1).long()
        else:
            true_labels_flat = true_labels.long()
    
        # --- 關鍵修改部分 ---
        # 使用 torch.topk 獲取前 2 高的機率和對應的索引
        top2_probs, top2_labels = torch.topk(probabilities, k=2, dim=-1)
        predicted_labels = top2_labels[:, 0]
        predicted_probs = top2_probs[:, 0]
        predicted_labels_2nd = top2_labels[:, 1]
        predicted_probs_2nd = top2_probs[:, 1]
        true_label_probs = torch.gather(probabilities, 1, true_labels_flat.unsqueeze(1)).squeeze(1)
        # print(f'predicted_labels.shape:{predicted_labels.shape},true_labels_flat.shape:{true_labels_flat.shape}')
        activate_cond1 = (predicted_labels != true_labels_flat)
        activate_cond2 = (predicted_labels == true_labels_flat) & (predicted_probs < (predicted_probs_2nd * 3))
        activate_map = activate_cond1 | activate_cond2
        
        i_tensor = torch.full_like(predicted_labels, fill_value=i, dtype=torch.float)
        result_tensor = torch.stack([
            i_tensor,
            predicted_labels.float(),
            predicted_probs,
            predicted_labels_2nd.float(),  # 新增
            predicted_probs_2nd,           # 新增
            true_labels_flat.float(),
            true_label_probs,
            activate_map.float()
        ], dim=1)
        
        return top_ks.tolist(),result_tensor,activate_map

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
        classifier_stats_in_process = []
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
            candidate,clasifier_stats,activate_map = self.calculate_candidate(x_cur,t_cur,labels,i)
            classifier_stats_in_process.append(clasifier_stats)
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
            d_cur = (x_hat - self.denoise_with_candidate(x_hat, t_hat, labels, i,seed,candidate,activate_map)) / t_hat
            if torch.isnan(d_cur).any():
                debug_print(self.debug,f"step {i}: d_cur is NaN")
            x_next = x_hat + (t_next - t_hat) * d_cur
            if torch.isnan(x_next).any():
                debug_print(self.debug,f"step {i}: x_next is NaN")

            # Apply 2nd order correction.
            debug_print(self.debug,f"step {i} Huen")
            if i < self.num_steps - 1:
                if self.heun_guid:
                    d_prime = (x_next - self.denoise_with_candidate(x_next, t_next, labels, i+1,seed,candidate,activate_map)) / t_next
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


        classifier_stats_all = torch.stack(classifier_stats_in_process,axis=0)
        classifier_stats_all = classifier_stats_all.permute(1,0,2)
        if self.pca is not None:
            # There should be T=num_steps+1 trajactory
            assert len(pca_features_in_process)==self.num_steps + 1
            # For pca features
            # List[T of [B, C]] → [T, B, C]
            # Tranpose to [B, T, C]
            return x_next,np.transpose(np.stack(pca_features_in_process, axis=0), (1, 0, 2)),classifier_stats_all
        else:
            return x_next,None,classifier_stats_all
    def denoise_with_candidate(self, x: torch.Tensor, sigma: torch.Tensor, labels: torch.Tensor, i: int,seed: torch.Tensor,candidates:list,activate_map):
        # t : noise scale sigma at time i
        # i : iterate from 0,1,...,num_steps-1
        # num_steps: N in EDM2
        if torch.isnan(x).any(): debug_print(self.debug,"x still NaN before net()")
        if torch.isnan(sigma).any(): debug_print(self.debug,"sigma still NaN before net()")
        debug_print(self.debug,f"In denoise_with_candidate sigma:{sigma}")
        Dx = self.net(x, sigma, labels).to(self.dtype)
        if not isinstance(Dx, torch.Tensor):
            debug_print(self.debug,f"Dx type is {type(Dx)}")
        if torch.isnan(Dx).any(): print("Dx is NaN!")
        # Compute scheduled guidance scale
        if self.guidance_scheduler:
            # debug_print(self.debug,f"guidance_scheduler: {self.guidance_scheduler}")
            scheduled_guidance = self.guidance_scheduler(i, self.num_steps, self.guidance,t=sigma,**self.sampler_kwargs)
        else:
            scheduled_guidance = self.guidance
        debug_print(self.debug,f"scheduled_guidance: {scheduled_guidance}")
        if scheduled_guidance == 1:
            return Dx
        
        ref_Dx = self.gnet(x, sigma, labels).to(self.dtype)
        
        return ref_Dx.lerp(Dx, scheduled_guidance)