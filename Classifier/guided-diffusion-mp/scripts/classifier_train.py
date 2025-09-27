"""
Train a noised image classifier on ImageNet.
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.append('../guided-diffusion')

import blobfile as bf
import torch as th
# import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.optim import AdamW

from guided_diffusion import dist_util, logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)
from guided_diffusion.train_util import parse_resume_step_from_filename, log_loss_dict

from base.torch_utils import distributed as dist
import time
from base.torch_utils import misc
import torch
import base.dnnlib as dnnlib
import string

def main():
    # mp.set_start_method("spawn")
    args = create_argparser().parse_args()
    dist.print0(f"args.noised={args.noised}")
    dist.print0(f"args.alpha={args.alpha}")
    dist.print0(f"args.classifier_in_channels={args.classifier_in_channels}")
    dist.print0(f"args.max_grad_norm={args.max_grad_norm}")
    # local_rank = int(os.environ["LOCAL_RANK"])
    # th.cuda.set_device(local_rank)

    # # Init process group
    # backend = "nccl" if th.cuda.is_available() else "gloo"
    # dist.init_process_group(backend=backend, init_method="env://")
    
    # logger.configure()
    # Initialize.
    seed = 0
    cudnn_benchmark     = True # Enable torch.backends.cudnn.benchmark?
    batch_size = args.batch_size
    batch_gpu           = None     # Limit batch size per GPU. None = no limit.
    total_nimg          = 8<<30    # Train for a total of N training images.
    slice_nimg          = None     # Train for a maximum of N training images in one invocation. None = no limit.
    status_nimg         = 128<<10  # Report status every N training images. None = disable.
    snapshot_nimg       = 8<<20    # Save network snapshot every N training images. None = disable.
    checkpoint_nimg     = 128<<20  # Save state checkpoint every N training images. None = disable.
    train_dataset_kwargs      = dict(class_name='training.dataset.ImageFolderDataset',
         path=args.data_dir)
    valid_dataset_kwargs      = dict(class_name='training.dataset.ImageFolderDataset',
         path=args.val_data_dir)
    if args.classifier_in_channels==4:
        encoder_kwargs      = dict(class_name='training.encoders.StabilityVAEEncoder')
    else:
        assert args.classifier_in_channels==3
        encoder_kwargs      = dict(class_name='training.encoders.StandardRGBEncoder')
       
    data_loader_kwargs  = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=2, prefetch_factor=2)
    device              = torch.device('cuda')
    state = dnnlib.EasyDict(cur_nimg=0, total_elapsed_time=0)
    
    prev_status_time = time.time()
    misc.set_random_seed(seed, dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Validate batch size.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    assert total_nimg % batch_size == 0
    assert slice_nimg is None or slice_nimg % batch_size == 0
    assert status_nimg is None or status_nimg % batch_size == 0
    assert snapshot_nimg is None or (snapshot_nimg % batch_size == 0 and snapshot_nimg % 1024 == 0)
    assert checkpoint_nimg is None or (checkpoint_nimg % batch_size == 0 and checkpoint_nimg % 1024 == 0)
    
    dist.print0('Initializing dist...')
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    assert torch.distributed.is_initialized(), "Distributed backend not initialized!"
    print(f"[Rank {torch.distributed.get_rank()}] Backend: {torch.distributed.get_backend()}")
    

    # Setup dataset, encoder, and network.
    dist.print0('Loading dataset...')
    train_dataset_obj = dnnlib.util.construct_class_by_name(**train_dataset_kwargs)
    valid_dataset_obj = dnnlib.util.construct_class_by_name(**valid_dataset_kwargs)
    ref_image, ref_label = train_dataset_obj[0]
    ref_image, ref_label = valid_dataset_obj[0]

    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    ref_image = encoder.encode_latents(torch.as_tensor(ref_image).to(device).unsqueeze(0))
    

    # logger.log(f"creating model and diffusion, rank = {dist.get_rank()}...")
    all_parameters = args_to_dict(args, classifier_and_diffusion_defaults().keys())
    print(all_parameters)
    model, diffusion = create_classifier_and_diffusion(
        **all_parameters
    )
    # logger.log(f"moving model to GPU, rank = {dist.get_rank()}...")
    model.to(dist_util.dev())
    # logger.log(f"prepare to run parse_resume_step_from_filename, rank = {dist.get_rank()}...")
    resume_step = 0
    if args.resume_checkpoint:
        resume_step = parse_resume_step_from_filename(args.resume_checkpoint)
        # logger.log(f"prepare to run load_state_dict, rank = {dist.get_rank()}...")
        # logger.log(f"loading model from checkpoint: {args.resume_checkpoint}... at {resume_step} step")
        model.load_state_dict(
            dist_util.load_state_dict(
                args.resume_checkpoint, map_location=dist_util.dev()
            )
        )
    return
    # logger.log(f"Before running dist_util.sync_params(model.parameters()), rank = {dist.get_rank()}...")
        
    # Needed for creating correct EMAs and fp16 parameters.
    dist_util.sync_params(model.parameters())
    # logger.log(f"After running dist_util.sync_params(model.parameters()), rank = {dist.get_rank()}...")
     

    mp_trainer = MixedPrecisionTrainer(
        model=model, use_fp16=args.classifier_use_fp16, initial_lg_loss_scale=16.0
    )

    model = DDP(
        model,
        device_ids=[dist_util.dev()],
        output_device=dist_util.dev(),
        broadcast_buffers=False,
        bucket_cap_mb=128,
        find_unused_parameters=False,
    )

    # logger.log("creating data loader...")
    # data = load_data(
    #     data_dir=args.data_dir,
    #     batch_size=args.batch_size,
    #     image_size=args.image_size,
    #     class_cond=True,
    #     random_crop=True,
    # )
    # if args.val_data_dir:
    #     val_data = load_data(
    #         data_dir=args.val_data_dir,
    #         batch_size=args.batch_size,
    #         image_size=args.image_size,
    #         class_cond=True,
    #     )
    # else:
    #     val_data = None
    dist.print0('Creating dataset iterator...')
    train_dataset_sampler = misc.InfiniteSampler(dataset=train_dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=state.cur_nimg)
    train_dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=train_dataset_obj, sampler=train_dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    valid_dataset_sampler = misc.InfiniteSampler(dataset=valid_dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=state.cur_nimg)
    valid_dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=valid_dataset_obj, sampler=valid_dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    
    #================================================

    dist.print0(f"creating optimizer...")
    opt = AdamW(mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_checkpoint:
        opt_checkpoint = bf.join(
            bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt"
        )
        logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
        opt.load_state_dict(
            dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev())
        )

    dist.print0("training classifier model...")

    def forward_backward_log(data_loader, prefix="train"):
        # batch, extra = next(data_loader)
        # labels = extra["y"].to(dist_util.dev())
        #-------------------------------上面原本下面抄edm2過來
        # images, labels = next(dataset_iterator)
        batch, labels = next(data_loader)
        labels = labels.to(dist_util.dev())
        batch = encoder.encode_latents(batch.to(device))
        # print(labels.shape)
        #-------------------
        


        batch = batch.to(dist_util.dev())
        # Noisy images
        if args.noised:
            # t = th.rand((batch.shape[0]), device=dist_util.dev()) * 79.998 + 0.002 # EDM2: a timestep is uniformly chosen between sigma_min and sigma_max
            alpha=args.alpha
            beta=2.0
            uniform_prob=args.uniform_prob
            low=0.002
            high=80.0
            beta_dist = torch.distributions.Beta(alpha, beta)
            beta_samples = beta_dist.sample((batch_size,)).to(device)
            use_uniform = torch.rand(batch_size, device=device) < uniform_prob
            uniform_samples = torch.rand(batch_size, device=device)
            samples = torch.where(use_uniform, uniform_samples, beta_samples)
            t = low + samples * (high - low)
            if prefix=='train':
                batch = batch + th.randn_like(batch, device=dist_util.dev()) * t.view(-1, 1, 1, 1) # x_0 + sigma(aka t) * N(0,I)
            else:#prefix=='val'
                batch = batch
        else:
            t = th.rand((batch.shape[0]), device=dist_util.dev()) * 0 + 0.002 # EDM2: a timestep is uniformly chosen between sigma_min and sigma_max
            batch = batch# + th.randn_like(batch, device=dist_util.dev()) * t.view(-1, 1, 1, 1) # x_0 + sigma(aka t) * N(0,I)
        
        for i, (sub_batch, sub_labels, sub_t) in enumerate(
            split_microbatches(args.microbatch, batch, labels, t)
        ):
            if prefix=='train':
                logits = model(sub_batch, timesteps=sub_t)
                loss = F.cross_entropy(logits, sub_labels, reduction="none")
                
                losses = {}
                
                losses[f"{prefix}_loss"] = loss.detach()
                losses[f"{prefix}_acc@1"] = compute_top_k(
                    logits, sub_labels, k=1, reduction="none"
                )
                losses[f"{prefix}_acc@5"] = compute_top_k(
                    logits, sub_labels, k=5, reduction="none"
                )
            else:#prefix=='val'
                losses = {}
                num_steps = 32
                test_t_list = [0.002]
                num_steps = 32
                sigma_max = 80
                sigma_min = 0.002
                rho = 7
                
                step_indices = torch.arange(num_steps)
                
                t_steps = (
                    sigma_max ** (1 / rho)
                    + step_indices / (num_steps - 1)
                      * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
                ) ** rho
                
                # 最後一個 step 設為 0
                t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
                for t in reversed(t_steps[:-1:2]):
                    test_t_list.append(t)
                for idx,t_val in enumerate(test_t_list):
                    t_to_test = th.full((sub_batch.shape[0],), t_val, device=device)  # 建立所有值都為 t_val 的 tensor
                    noised_sub_batch = sub_batch + th.randn_like(sub_batch) * t_to_test.view(-1, 1, 1, 1) 
                    logits = model(noised_sub_batch, timesteps=t_to_test)
                    loss = F.cross_entropy(logits, sub_labels, reduction="none")
                    letter = string.ascii_lowercase[idx]  # 0 -> 'a', 1 -> 'b', ...
                    losses[f"{prefix}_{letter}_t={t_val:.3f}_loss"] = loss.detach()
                    losses[f"{prefix}_{letter}_t={t_val:.3f}_acc@1"] = compute_top_k(
                        logits, sub_labels, k=1, reduction="none"
                    )
                    losses[f"{prefix}_{letter}_t={t_val:.3f}_acc@5"] = compute_top_k(
                        logits, sub_labels, k=5, reduction="none"
                    )
                t_to_test = th.rand((sub_batch.shape[0]), device=dist_util.dev()) * 79.998 + 0.002 # EDM2: a timestep is uniformly chosen between sigma_min and sigma_max
                noised_sub_batch = sub_batch + th.randn_like(sub_batch) * t_to_test.view(-1, 1, 1, 1) 
                logits = model(noised_sub_batch, timesteps=t_to_test)
                loss = F.cross_entropy(logits, sub_labels, reduction="none")
                losses[f"{prefix}_uniform_loss"] = loss.detach()
                losses[f"{prefix}_uniform_acc@1"] = compute_top_k(
                    logits, sub_labels, k=1, reduction="none"
                )
                losses[f"{prefix}_uniform_acc@5"] = compute_top_k(
                    logits, sub_labels, k=5, reduction="none"
                )
            log_loss_dict(diffusion, sub_t, losses) #為啥要計算diffusion model loss又沒用到
            del losses
            loss = loss.mean()
            if loss.requires_grad:
                if i == 0:
                    mp_trainer.zero_grad()
                mp_trainer.backward(loss * len(sub_batch) / len(batch))

    for step in range(args.iterations - resume_step):
        logger.logkv("step", step + resume_step)
        logger.logkv(
            "samples",
            (step + resume_step + 1) * args.batch_size * dist.get_world_size(),
        )
        if args.anneal_lr:
            set_annealed_lr(opt, args.lr, (step + resume_step) / args.iterations)
        forward_backward_log(train_dataset_iterator)
        # 在呼叫 optimize() 之前，進行梯度裁剪
        if args.max_grad_norm is not None:
            # 這裡我們直接對 master_params 進行裁剪
            # MixedPrecisionTrainer 已經處理了梯度從模型到 master_params 的複製
            torch.nn.utils.clip_grad_norm_(mp_trainer.master_params, max_norm=args.max_grad_norm)
           
        mp_trainer.optimize(opt)
        if valid_dataset_iterator is not None and not step % args.eval_interval and dist.get_rank() == 0:
            with th.no_grad():
                with model.no_sync():
                    model.eval()
                    forward_backward_log(valid_dataset_iterator, prefix="val")
                    model.train()
        if not step % args.log_interval:
            print(datetime.now())
            logger.dumpkvs()
        if (
            step
            and dist.get_rank() == 0
            and not (step + resume_step) % args.save_interval
        ):
            logger.log("saving model...")
            save_model(mp_trainer, opt, step + resume_step)

    if dist.get_rank() == 0:
        logger.log("saving model...")
        save_model(mp_trainer, opt, step + resume_step)
    dist.barrier()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model(mp_trainer, opt, step):
    if dist.get_rank() == 0:
        th.save(
            mp_trainer.master_params_to_state_dict(mp_trainer.master_params),
            os.path.join(logger.get_dir(), f"model{step:06d}.pt"),
        )
        th.save(opt.state_dict(), os.path.join(logger.get_dir(), f"opt{step:06d}.pt"))


def compute_top_k(logits, labels, k, reduction="mean"):
    # print('Before reshape',labels.shape)
    if labels.ndim == 2 and labels.shape[-1] > 1:
        # 假設是 one-hot，轉成 class index
        labels = labels.argmax(dim=-1)

    _, top_ks = th.topk(logits, k, dim=-1)
    # print('After reshape',labels.shape)
    # print(top_ks.shape,labels[:, None].shape)
    correct = (top_ks == labels[:, None]).float().sum(dim=-1)

    if reduction == "mean":
        return correct.mean().item()
    elif reduction == "none":
        return correct


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        eval_interval=5,
        save_interval=10000,
        alpha=0.3,
        uniform_prob = 0.001,
        classifier_in_channels=4,
        max_grad_norm = 1.0,
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
