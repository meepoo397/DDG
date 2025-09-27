import torch
import numpy as np
import tqdm
from typing import Callable,Optional

from base.training.encoders import Encoder
from utils import debug_print
from sampler.scheduler import guidance_scheduler_config
from diffusion import create_diffusion

Network = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
Classifier = Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
class Sampler:
    def __init__(
            self, net: Callable, encoder: Callable,
            classifier:Callable,
            num_steps = 32, guidance_scheduler = 'const_scheduler', guidance = 1.7,
            dtype = torch.float32, sigma_min = 0.002, sigma_max = 80, rho = 7,
            S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, randn_like=torch.randn_like,
            pca = None, debug = False,
            **sampler_kwargs,
        ) -> None:
        self.net = net
        self.sample_fn = self.net.forward_with_cfg
        self.encoder = encoder
        self.classifier = classifier
        self.num_steps = num_steps
        if guidance_scheduler not in guidance_scheduler_config:
            raise KeyError(f"Unknown guidance_scheduler: {guidance_scheduler}")
        self.guidance_scheduler = guidance_scheduler_config[guidance_scheduler]
        self.guidance = guidance
        self.dtype = dtype
        self.randn_like = randn_like
        self.debug = debug
        self.sampler_kwargs = sampler_kwargs
        self.heun_guid = sampler_kwargs.get('heun_guid', False)
        if 'heun_guid' not in sampler_kwargs:
            debug_print(self.debug,f"heun_guid is not specified using heun_guid=False")
        else:
            debug_print(self.debug,f"heun_guid={self.heun_guid}")
            
        # self.diffusion = create_diffusion(str(self.num_steps),noise_schedule="squaredcos_cap_v2")
        self.diffusion = create_diffusion(str(self.num_steps),noise_schedule="linear") 
        
    def calculate_candidate(self,x0:torch.Tensor,sigma:torch.Tensor,true_labels:torch.Tensor,i):
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
        logits = self.classifier(xo_predicted,dummy_t)
        ## 3. x0 prediction with x input
        # logits = self.classifier(x)
        log_proba = torch.log_softmax(logits, dim=-1)
        true_labels_flat = true_labels.argmax(dim=-1).long()
        debug_print(self.debug,f'true_labels_flat.shape:{true_labels_flat.shape}')
        debug_print(self.debug,f'true_labels_flat:{true_labels_flat}')
        # debug_print(self.debug,f'log_proba before -inf self index:{log_proba}')
        _, top_ks_for_check_correct = torch.topk(log_proba, self.n, dim=-1)
        correct = (top_ks_for_check_correct == true_labels_flat[:, None]).float().sum(dim=-1).mean().item()
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
        average_true_prob = true_label_probs.mean().item()
        diff_average_true_prob = average_true_prob - prev_average_true_prob
        # print(f'Step {i} diff_average_true_prob:{diff_average_true_prob}')
        # print(f'predicted_labels.shape:{predicted_labels.shape},true_labels_flat.shape:{true_labels_flat.shape}')
        
        average_true_prob_batch = true_label_probs.mean().item()
        debug_print(self.debug,f'Average true label probability for the batch: {average_true_prob_batch}')
        device = adaptive_scheduler_state.device
        predicted_probs = predicted_probs.to(device)
        predicted_probs_2nd = predicted_probs_2nd.to(device)
        true_labels_flat = true_labels_flat.to(device)
        predicted_labels = predicted_labels.to(device)

        w_t =  (1 + self.balance_coeff * diff_average_true_prob) if diff_average_true_prob > self.open_prob else 1
        
        
        
        
        i_tensor = torch.full_like(predicted_labels, fill_value=i, dtype=torch.float)
        top10_probs, top10_labels = torch.topk(probabilities, k=10, dim=-1)

        state = adaptive_scheduler_state[:,0]
        stats_tensor = torch.stack([
            torch.full_like(true_label_probs, fill_value=i, dtype=torch.float).to('cpu'), # 1
            true_labels_flat.float().to('cpu'),# 1
            true_label_probs.to('cpu'),# 1
            state.float().to('cpu')# 1
        ], dim=1)
    
        # 將 Top-10 索引和機率張量與統計數據張量連接起來
        result_tensor = torch.cat((stats_tensor, top10_labels.float().to('cpu'), top10_probs.to('cpu'), probabilities.to('cpu')), dim=1)

        
        return top_ks.tolist(),result_tensor,w_t,average_true_prob
    def sample(self, noise: torch.Tensor, y: torch.Tensor, seeds: list):
        device = noise.device
        n = noise.shape[0]
        y_null = torch.tensor([1000] * n, device=device)
        print(f'y.shape:{y.shape},y_null.shape:{y_null.shape}')
        y = torch.cat([y, y_null], 0)
        
        
        indices = list(range(self.num_steps))[::-1]

        x = torch.cat([noise, noise], 0)
        # uj in Appendix A
        M, N, C1, C2 = 1000, 250, 0.001, 0.008
        u_array = [0 for i in range(M+1)]
        
        def alpha_bar(j):
            return np.sin(np.pi / 2 * j / (M * (C2 + 1))) ** 2
        
        for j in range(M-1, -1, -1):
            alpha_bar_j = alpha_bar(j)
            alpha_bar_j_plus_1 = alpha_bar(j + 1)
            u_array[j] = np.sqrt((u_array[j + 1] ** 2 + 1) / max(alpha_bar_j / alpha_bar_j_plus_1, C1) - 1)
        sigma = [u_array[i] for i in indices]
        for i in indices:
            t = torch.tensor([i] * n * 2, device=device)
            # Compute sigma in DDPM for sigma interval
            # Please reference Appendix A in "Applying Guidance in a Limited Interval ImprovesSample and Distribution Quality in Diffusion Models"
            
            # uj = D
            sigma_i = sigma[i]
            print(f'step:{i},sigma:{sigma}')
            # print(self.diffusion.alphas_cumprod)
            w_t = self.guidance_scheduler(i, self.num_steps, self.guidance,sigma)
            
            # candidate,clasifier_stat,w_t_fake,average_true_prob = self.calculate_candidate(self,x,sigma,y,i)
            model_kwargs = dict(y=y, cfg_scale=w_t)
            with torch.no_grad():
                out = self.diffusion.p_sample(
                    self.sample_fn,
                    x,
                    t,
                    clip_denoised=False,
                    denoised_fn=None,
                    cond_fn=None,
                    model_kwargs=model_kwargs,
                )
                img = out["sample"]
        return x
