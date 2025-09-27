# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the given model."""

import os
import re
import string
from telnetlib import TELNET_PORT
import warnings
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import base.dnnlib
from base.torch_utils import distributed as dist
import numpy as np
import torch
from base.torch_utils import distributed as dist
from edm2impl.utils import debug_print
from edm2impl.sampler_run import Sampler, Network

class Time_Offset_Sampler(Sampler):
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
        if 'delta' in sampler_kwargs:
            self.delta = sampler_kwargs['delta']
            debug_print(self.debug,f"Using input delta={self.delta} in Time offset sampling")
        else:
            debug_print(self.debug,"[Warning] 'delta' not found in sampler_kwargs, using default delta = 0")
            self.delta = 0 
        if 'use_gnet' in sampler_kwargs:
            self.use_gnet = sampler_kwargs['use_gnet']
            debug_print(self.debug,f"use_gnet={self.use_gnet}")
        else:
            self.use_gnet = False
            debug_print(self.debug,f"use_gnet is not specified using net instead")
        self.heun_tog = sampler_kwargs.get('heun_guid', False)
        if 'heun_guid' not in sampler_kwargs:
            debug_print(self.debug,f"heun_guid is not specified using heun_guid=False")
        else:
            debug_print(self.debug,f"heun_guid={self.heun_tog}")
        self.delta_scheduler = sampler_kwargs.get('delta_scheduler', 'const_scheduler')
    def collect_pca_projection(self,x:torch.Tensor,pca_features_in_process:list):
        if self.pca:
            flattened = x.detach().to("cpu").view(x.size(0), -1).numpy()
            pca_features_in_process.append(self.pca.transform(flattened))
    def nan_check(self,x:torch.Tensor,i,name='unnamed tensor'):
        if torch.isnan(x).any():
            debug_print(self.debug,f"step {i}: {name} is NaN")
    def sample(self, noise: torch.Tensor, labels: torch.Tensor,seeds:list):
        return self.sample_ode(noise, labels,seeds)
    
    def sample_ode(self, noise: torch.Tensor, labels: torch.Tensor,seeds:list):
        # Reference of tuned algorithm is at https://hackmd.io/@jNjeFFL1RfK7FwSoxF4gAQ/H1dAy4aElg
        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=self.dtype, device=noise.device)
        # Rho schedule, note that sigma(t)=t in EDM2.
        t_steps = (self.sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([t_steps[:1],t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
        # List for pca projection(should collect numpy array of shape [B, C])
        pca_features_in_process = []
        # Main sampling loop.
        x_next = noise.to(self.dtype) * t_steps[0]
        self.collect_pca_projection(x_next,pca_features_in_process)
        for i, (t_prev,t_cur, t_next) in enumerate(zip(t_steps[0:-2],t_steps[1:-1], t_steps[2:])): # 0, ..., N-1
            debug_print(self.debug, f"step {i}/{self.num_steps}")
            self.nan_check(x_next,i,'x_next')
            x_cur = x_next

            # Increase noise temporarily.
            # Note that default parameter S_churn=0, may ignore this part. 
            if self.S_churn > 0 and self.S_min <= t_cur <= self.S_max:
                gamma = min(self.S_churn / self.num_steps, np.sqrt(2) - 1)
                t_hat = t_cur + gamma * t_cur
                x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * self.randn_like(x_cur)
            else:
                t_hat = t_cur
                x_hat = x_cur
            # Euler step.
            t_off = min(t_hat+self.delta_t(t_hat,i),t_prev)
            debug_print(self.debug,f"For Euler step t_prev:{t_prev},t_cur:{t_cur},t_next:{t_next},t_off:{t_off}")
            # d_i in new algorithm, i.e. dx/dt
            d_cur = self.drift_ode(x_hat,t_hat,t_off,labels,i)
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < self.num_steps - 1:
                t_off_prime = min(t_next+self.delta_t(t_next,i+1),t_cur)
                debug_print(self.debug,f"For Heun step t_next:{t_next},t_off_prime:{t_off_prime}")
                if self.heun_tog:
                    debug_print(self.debug,f"Using heun_tog true")
                    d_prime = self.drift_ode(x_next,t_next,t_off_prime,labels,i)
                else:
                    debug_print(self.debug,f"Using heun_tog false")
                    d_prime = self.drift_ode(x_next,t_next,t_next,labels,i)
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            self.collect_pca_projection(x_next,pca_features_in_process)
        # Return x_next,pca_features
        if self.pca is not None:
            # There should be T=num_steps+1 points(containing initial point)
            assert len(pca_features_in_process)==self.num_steps + 1
            # For pca features
            # List[T of [B, C]] → [T, B, C] → [B, T, C]
            return x_next,np.transpose(np.stack(pca_features_in_process, axis=0), (1, 0, 2))
        else:
            return x_next,None
    def drift_ode(self, x: torch.Tensor, t: torch.Tensor,t_off: torch.Tensor, labels: torch.Tensor, i: int):
        # Function for calculating dx/dt of Probabilty Flow ODE in tuned algorithm
        # t : note that in EDM2 sigma(t)=t
        # i     : iterate from 0,1,...,num_steps-1
        # num_steps: N=32 in EDM2
        Dx = self.net(x, t, labels).to(self.dtype)
        # Compute scheduled guidance scale
        scheduled_guidance = self.guidance_scheduler(i, self.num_steps, self.guidance,t=t,**self.sampler_kwargs)
        debug_print(self.debug,f"scheduled_guidance: {scheduled_guidance}")
        if scheduled_guidance == 1:
            return (x-Dx)/t # This is actually the predicted noise.
        if self.use_gnet:
            debug_print(self.debug, f"Using gnet")
            ref_Dx = self.gnet(x, t_off, labels).to(self.dtype)
        else:
            debug_print(self.debug, f"Using net not gnet")
            ref_Dx = self.net(x, t_off, labels).to(self.dtype)
        
        scale = (1 + (t_off - t) / t) ** 2
        debug_print(self.debug,f"t: {t},t_off:{t_off},scale:{scale}")
        s_main = (Dx-x)/(t**2)
        s_off = (ref_Dx-x)/(t_off**2)
        s_main_norm = torch.norm(s_main.view(s_main.shape[0], -1), dim=1)
        s_off_norm = torch.norm(s_off.view(s_off.shape[0], -1), dim=1)
        norm_alt = torch.minimum(s_main_norm/s_off_norm, torch.ones_like(s_main_norm)).view(-1, 1, 1, 1)
        debug_print(self.debug, f"L2 norm of s_main (per sample): {s_main_norm}")
        debug_print(self.debug, f"L2 norm of s_off (per sample): {s_off_norm}")
        return -t*s_main -t* (scheduled_guidance - 1)*(s_main - s_off)#*norm_alt)
    def delta_t(self,t,i):
        debug_print(self.debug, f"Using TOG with self.delta={self.delta}")
        if self.delta_scheduler=='const_scheduler':
            debug_print(self.debug, f"Using TOG with const delta_scheduler")
            return self.delta*t
        elif self.delta_scheduler=='linear_decrease_scheduler':
            weight = 2-(2*i)/self.num_steps
            debug_print(self.debug, f"Using TOG with linear decrease delta_scheduler, weight= {weight}")
            return self.delta*weight*t
        elif self.delta_scheduler=='S_scheduler':
            if i <3:
                weight = -0.5
            else:
                weight = 2-(2*i)/self.num_steps
            debug_print(self.debug, f"Using TOG with S_scheduler delta_scheduler, weight= {weight}")
            return self.delta*weight*t
        elif self.delta_scheduler=='linear_increase_scheduler':
            weight = (2*i)/self.num_steps
            debug_print(self.debug, f"Using TOG with linear decrease delta_scheduler, weight= {weight}")
            return self.delta*weight*t
        else:
            debug_print(self.debug, f"Using TOG with invalid delta_scheduler {self.delta_scheduler}")
            raise ValueError(f"Using TOG with invalid delta_scheduler {self.delta_scheduler}")
        
    def sample_sde(self, noise: torch.Tensor, labels: torch.Tensor,seeds):
        # Create seeds_array using seed for manual seeds in each batch.
        seeds_array = np.concatenate([
            np.random.default_rng(seed).integers(low=0, high=2**31, size=(self.num_steps,1))
            for seed in seeds], axis=1)
        # Time step discretization.
        step_indices = torch.arange(self.num_steps, dtype=self.dtype, device=noise.device)
        # Rho schedule, note that sigma(t)=t in EDM2.
        t_steps = (self.sigma_max ** (1 / self.rho) + step_indices / (self.num_steps - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho
        t_steps = torch.cat([t_steps[:1],t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
        # List for pca projection(should collect numpy array of shape [B, C])
        pca_features_in_process = []
        # Main sampling loop.
        x = noise.to(self.dtype) * t_steps[0]
        self.collect_pca_projection(x,pca_features_in_process)
        for i, (t_prev,t_cur, t_next,seed) in enumerate(zip(t_steps[0:-2],t_steps[1:-1], t_steps[2:],seeds_array)): # 0, ..., N-1
            self.nan_check(x,i,'x')
            debug_print(self.debug,f"For Euler step t_prev:{t_prev},t_cur:{t_cur},t_next:{t_next}")
            h = t_next - t_cur
            s_main = (self.net(x,t_cur,labels) - x)/(t_cur**2)
            t_off = min(t_cur + self.delta_t(t_cur,i), t_prev)
            # Euler Predictor.
            debug_print(self.debug, "Euler Start------------------")
            scheduled_guidance = self.guidance_scheduler(i, self.num_steps, self.guidance,t=t_cur,**self.sampler_kwargs)
            debug_print(self.debug, f"scheduled_guidance ={scheduled_guidance}")
            if self.use_gnet:
                debug_print(self.debug, f"Using gnet")
                s_off = (self.gnet(x,t_off,labels) - x)/(t_off**2)
            else:
                debug_print(self.debug, f"Using net not gnet")
                s_off = (self.net(x,t_off,labels) - x)/(t_off**2)
            s_main_norm = torch.norm(s_main.view(s_main.shape[0], -1), dim=1)
            s_off_norm = torch.norm(s_off.view(s_off.shape[0], -1), dim=1)
            norm_alt = torch.minimum(s_main_norm/s_off_norm, torch.ones_like(s_main_norm)).view(-1, 1, 1, 1)
            debug_print(self.debug, f"L2 norm of s_main (per sample): {s_main_norm}")
            debug_print(self.debug, f"L2 norm of s_off (per sample): {s_off_norm}")
            s_guided =  s_main + (scheduled_guidance - 1)*(s_main - s_off*norm_alt)
            g = torch.sqrt(2*t_cur) # obtain by EDM2 have sigma(t)=t,s(t)=1
            drift = -g**2*s_guided
            diffusion = g*torch.sqrt(-h)
            batch_size = x.shape[0]
            rand = torch.stack([
                torch.randn(
                    x[i].shape,
                    dtype=x.dtype,
                    device=x.device,
                    generator=torch.Generator(device=x.device).manual_seed(int(seed[i]))
                )
                for i in range(batch_size)
            ])
            x_euler = x + h*drift + diffusion * rand
            # Huen Corrector.
            debug_print(self.debug, "Huen Start------------------")
            if i < self.num_steps - 1:
                s_main_prime = (self.net(x_euler,t_next,labels) - x_euler)/(t_next**2)
                t_off_prime = min(t_next + self.delta_t(t_next,i+1), t_cur)
                scheduled_guidance = self.guidance_scheduler(i, self.num_steps, self.guidance,t_cur)
                if self.use_gnet:
                    s_off_prime = (self.gnet(x_euler,t_off_prime,labels) - x_euler)/(t_off_prime**2)
                else:
                    s_off_prime = (self.net(x_euler,t_off_prime,labels) - x_euler)/(t_off_prime**2)
                s_main_prime_norm = torch.norm(s_main_prime.view(s_main_prime.shape[0], -1), dim=1)
                s_off_prime_norm = torch.norm(s_off_prime.view(s_off_prime.shape[0], -1), dim=1)
                norm_prime_alt = torch.minimum(s_main_prime_norm/s_off_prime_norm,torch.ones_like(s_main_prime_norm)).view(-1, 1, 1, 1)
                debug_print(self.debug, f"L2 norm of s_main_prime_norm (per sample): {s_main_prime_norm}")
                debug_print(self.debug, f"L2 norm of s_off_prime_norm (per sample): {s_off_prime_norm}")
                
                s_guided_prime = s_main_prime + (scheduled_guidance - 1)*(s_main_prime - s_off_prime*norm_prime_alt)
                g_prime = torch.sqrt(2*t_next)
                drift_prime = - g_prime**2 * s_guided_prime
                drift_avg = 0.5 * (drift+drift_prime)
                x = x + h*drift_avg + diffusion * rand
            else:
                x = x_euler
            # PCA features transforming
            self.collect_pca_projection(x,pca_features_in_process)
        if self.pca is not None:
            # There should be T=num_steps+1 trajactory
            assert len(pca_features_in_process)==self.num_steps + 1
            # For pca features
            # List[T of [B, C]] → [T, B, C]
            # Tranpose to [B, T, C]
            return x,np.transpose(np.stack(pca_features_in_process, axis=0), (1, 0, 2))
        else:
            return x,None
    
