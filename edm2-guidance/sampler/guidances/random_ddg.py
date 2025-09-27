# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import numpy as np
import pandas as pd
import torch
from base.torch_utils import distributed as dist
from edm2impl.utils import debug_print
from edm2impl.sampler_run import Sampler,Network
import random
class Random_DDG_Sampler(Sampler):
    def __init__(
            self, net: Network, gnet: Network,
            num_steps = 32, guidance_scheduler = 'const_scheduler', guidance = 1.7,
            dtype = torch.float32, sigma_min = 0.002, sigma_max = 80, rho = 7,
            S_churn=0, S_min=0, S_max=float('inf'), S_noise=1, randn_like=torch.randn_like,
            pca = None, debug = False,
            **sampler_kwargs,
        ) -> None:
        super().__init__(
            net=net, gnet=gnet,
            num_steps=num_steps, guidance_scheduler=guidance_scheduler, guidance=guidance,
            dtype=dtype, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho,
            S_churn=S_churn, S_min=S_min, S_max=S_max, S_noise=S_noise,
            randn_like=randn_like, pca=pca, debug=debug,**sampler_kwargs,
        )
        negative_class_csv_path = sampler_kwargs.get('negative_class_csv_path', './plot/negative_class.csv')
        full_random = sampler_kwargs.get('full_random', True)  # 是否 Random 1000個class
        self.k = sampler_kwargs.get('k_average', 1)  # 平均n個score
        self.num_classes = sampler_kwargs.get('num_classes', 1000)

        if not full_random:
            self.full_random=False
            self.n = sampler_kwargs.get('n_random', 3)  # 若需要多個負樣本，可設 top_k > 1
            try:
                df = pd.read_csv(negative_class_csv_path, sep=None, engine='python')
                debug_print(debug, f"df.columns:{df.columns}")
                required_columns = ['Class ID', 'Similar Class ID', 'Similarity Score']
                for col in required_columns:
                    if col not in df.columns:
                        raise ValueError(f"缺少必要欄位：{col}")
            
                grouped = df.groupby('Class ID')
            
                if self.n == 1:
                    # 選擇 Similarity Score 最高的
                    self.negative_label_map = grouped.apply(
                        lambda g: g.sort_values('Similarity Score', ascending=False).iloc[0]['Similar Class ID']
                    ).to_dict()
                else:
                    # 選擇 Similarity Score 前 n_candidate 高的
                    self.negative_label_map = grouped.apply(
                        lambda g: g.sort_values('Similarity Score', ascending=False).head(self.n)\
                        ['Similar Class ID'].tolist()).to_dict()
                debug_print(debug, f"✅ 成功建立 negative_label_map（依 Similarity Score 排序），共{len(self.negative_label_map)} 個類別")
                debug_print(debug, f"negative_label_map:{self.negative_label_map }")
            except Exception as e:
                print(f"❌ 無法建立 negative_label_map：{e}")
                self.negative_label_map = {}
        else:
            self.full_random=True
            self.negative_label_map = {label:list(range(1,self.num_classes)) for label in list(range(0,self.num_classes))}
            self.n = sampler_kwargs.get('n_random', 1000)  # 若需要多個負樣本，可設 top_k > 1
        
        
    def denoise(self, x: torch.Tensor, sigma: torch.Tensor, labels: torch.Tensor, i: int,seed: torch.Tensor):
        # t : noise scale sigma at time i
        # i : iterate from 0,1,...,num_steps-1
        # num_steps: N in EDM2
        if torch.isnan(x).any(): debug_print(self.debug,"x still NaN before net()")
        if torch.isnan(sigma).any(): debug_print(self.debug,"sigma still NaN before net()")
        debug_print(self.debug,f"In denoise sigma:{sigma}")
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
        # Random pick a different class label for guidance network
        batch_size, num_classes = labels.shape
        label_indices = labels.argmax(dim=1)
        debug_print(self.debug,f"Using random negative ddg, original class {label_indices}")
        # random_offsets = torch.randint(1, num_classes, size=[batch_size], device=labels.device)
        # random_offsets = torch.randint(1, 2, size=[batch_size], device=labels.device)
        # negative_indices = (label_indices + random_offsets) % num_classes
        # negative_labels = torch.zeros_like(labels)
        # negative_labels[torch.arange(batch_size), negative_indices] = 1
        # 隨機抓 n 個 negative labels，生成多個 ref_Dx 後平均
        n = self.n
        k = self.k
        debug_print(self.debug,f"Using random negative ddg, n:{n},k:{k}")
        seed = int(seed[0])
        debug_print(self.debug,f"Using seed:{seed}")
        random.seed(seed)

        ref_Dx_list = []

        # 對 batch 中每個 sample 建立 n 個 negative label
        negative_label_batches = []
        label_indices_list = label_indices.tolist()

        for idx in label_indices_list:
            debug_print(self.debug,f"Using random negative ddg, idx: {idx}")
            candidates = self.negative_label_map.get(idx, [(idx + 1) % num_classes])
            # debug_print(self.debug,f"Using random negative ddg, candidates {candidates}")
        
            # 保證 candidates 是 list，並取前 n 個
            if not isinstance(candidates, list):
                candidates = [candidates]

            candidates = candidates[:n]  # 只取前 n 個
            debug_print(self.debug,f"Using random negative ddg, len of candidates {len(candidates)}")
            if len(candidates) <10:
                debug_print(self.debug,f"Using random negative ddg, candidates {candidates}")
        
            # 若候選不足 k 則重複抽樣 確保抽到剛好k個來準備做k_average
            if len(candidates) >= n:
                debug_print(self.debug,f"len(candidates) >= k")
        
                sampled = random.sample(candidates, k)
                if self.full_random:
                    assert len(candidates)==self.num_classes-1
                    sampled = [(idx+sample) % self.num_classes for sample in sampled]
            else:
                debug_print(self.debug,f"len(candidates) < k")
                sampled = random.choices(candidates, k=k)
                if self.full_random:
                    assert len(candidates)==self.num_classes-1
                    sampled = [(idx+sample) % self.num_classes for sample in sampled]

            negative_label_batches.append(sampled)  # 每個 sample 對應 n 個 neg label
        debug_print(self.debug,f"Using random negative ddg, negative class {negative_label_batches}")
        # 將每個 sample 的第 j 個負 label 拿出，構成 n 組 batch_size 長度的 label set
        for j in range(k):
            neg_labels = torch.zeros_like(labels)
            neg_class_indices = [negative_label_batches[i][j] for i in range(batch_size)]
            neg_labels[torch.arange(batch_size), torch.tensor(neg_class_indices, device=labels.device)] = 1
            ref_Dx_list.append(self.net(x, sigma, neg_labels).to(self.dtype))

        ref_Dx = torch.stack(ref_Dx_list, dim=0).mean(dim=0)

        if self.debug:
            s_main_norm = torch.norm(((Dx-x)/(sigma**2)).view(x.shape[0], -1), dim=1)
            s_negative_norm = torch.norm(((ref_Dx-x)/(sigma**2)).view(x.shape[0], -1), dim=1)
            debug_print(self.debug, f"L2 norm of s_main_norm (per sample): {s_main_norm}")
            debug_print(self.debug, f"L2 norm of s_negative_norm (per sample): {s_negative_norm}")
        
        return ref_Dx.lerp(Dx, scheduled_guidance)