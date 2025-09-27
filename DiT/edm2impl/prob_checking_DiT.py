import os
import torch
import numpy as np
import joblib
import PIL.Image
import click
import re

import base.dnnlib as dnnlib
from base.generate_images import StackedRandomGenerator
from base.torch_utils import distributed as dist

from sampler.guidance import guidance_config
from plot_trajectories import plot_all_trajectories
from utils import debug_print
import tqdm
import pickle
from classifier.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
    create_classifier,
    classifier_defaults,
)
# from classifier.res50_train import create_resnet50_classifier
from classifier.unet import EncoderUNetModel
import torchvision.models as models
import torchvision.transforms as transforms

import types
import copy

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_XL_2
def generate_images_complete(
    net_path,                                        # Main network.
    outdir              = None,                 # Where to save the output images. None = do not save.
    subdirs             = False,                # Create subdirectory for every 1000 seeds?
    seeds               = range(16, 24),        # List of random seeds.
    max_batch_size      = 32,                   # Maximum batch size for the diffusion model.
    encoder_batch_size  = 4,                    # Maximum batch size for the encoder. None = default.
    verbose             = True,                 # Enable status prints?
    device              = torch.device("cuda"), # torch.device('cuda'), # Which compute device to use.
    ############################################################
    # Additional parameters
    guidance_scheduler  = 'const_scheduler',    # Guidance scale scheduler, default constant scheduling.
    debug               = False,                # Debug flag for NaN message.
    pca_plot            = False,                # Flag to draw pca plot .
    pca_path            = './plot/ipca_model.pkl',   # Path of ipca model fitted by Imagnet512 2012 validation set
    classifier_path     = None,                 # Path of classifier model .pt file
    pca=None,
    ##############################################################
    sampler             = 'cfg',                # Name of the sampler.
    **sampler_kwargs,                           # Additional arguments for the sampler function.
):
    
    image_size = 512 #@param [256, 512]
    vae_model = "stabilityai/sd-vae-ft-ema" #@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
    latent_size = int(image_size) // 8
    # Load model:
    model = DiT_XL_2(input_size=latent_size).to(device)
    state_dict = find_model(net_path) if net_path is not None else find_model("DiT-XL-2-512x512.pt")
    model.load_state_dict(state_dict)
    model.eval() # important!
    vae = AutoencoderKL.from_pretrained(vae_model).to(device)
    # Load classifier(if using sampler related with classifier)
    classifier=None
    if sampler=='map_neg_ddg' or sampler=='cg' or sampler=='map_x0pred':
        assert classifier_path is not None
    if classifier_path is not None:
        num_classes = 1000
        in_channels = 4
        if classifier_path == '/data/guidance-team-new/classifier_log_trained_by_train_res50_10/checkpoint_epoch7_batch90000.pt': #cl_10
            print(f'Successfully match classifier path and structure:cl_10,{classifier_path}')
            classifier = EncoderUNetModel(
                image_size=64,
                in_channels=4,
                model_channels=128,
                out_channels=1000,
                num_res_blocks=4,
                attention_resolutions=(2, 4, 8),
                channel_mult=(1, 2, 3, 4),
                use_fp16=False,
                num_head_channels=64,
                use_scale_shift_norm=True,
                resblock_updown=True,
                pool='attention')
            checkpoint = torch.load(classifier_path, map_location=device)
            device='cuda:0'
            classifier.to(device)
            print(f"Loading checkpoint from {classifier_path}...")
            classifier.load_state_dict(checkpoint['model_state_dict'])
            classifier.eval()
        ###### cl_10
        # cl_9
        elif classifier_path=='/data/guidance-team-new/classifier_log_trained_by_train_res50_9/checkpoint_epoch17_batch180000.pt':
            print(f'Successfully match classifier path and structure:cl_9,{classifier_path}')
            classifier = EncoderUNetModel(
                image_size=64,
                in_channels=in_channels,
                model_channels=128,
                out_channels=1000,
                num_res_blocks=2,
                attention_resolutions=(32,16,8),
                channel_mult=(1, 2, 3, 4),
                use_fp16=False,
                num_head_channels=64,
                use_scale_shift_norm=True,
                resblock_updown=True,
                pool="attention",
            )
            checkpoint = torch.load(classifier_path, map_location=device)
            device='cuda:0'
            classifier.to(device)
            print(f"Loading checkpoint from {classifier_path}...")
            classifier.load_state_dict(checkpoint['model_state_dict'])
            classifier.eval()
        elif classifier_path == '/data/guidance-team-new/classifier_log_dit_trained_by_train_res50/checkpoint_epoch16_batch170000.pt':
            print(f'Successfully match classifier path and structure:cl_9,{classifier_path}, trained for DiT')
            classifier = EncoderUNetModel(
                image_size=64,
                in_channels=in_channels,
                model_channels=128,
                out_channels=1000,
                num_res_blocks=2,
                attention_resolutions=(32,16,8),
                channel_mult=(1, 2, 3, 4),
                use_fp16=False,
                num_head_channels=64,
                use_scale_shift_norm=True,
                resblock_updown=True,
                pool="attention",
            )
            checkpoint = torch.load(classifier_path, map_location=device)
            device='cuda:0'
            classifier.to(device)
            print(f"Loading checkpoint from {classifier_path}...")
            classifier.load_state_dict(checkpoint['model_state_dict'])
            classifier.eval()
        elif classifier_path=='/data/guidance-team-new/Imagenet64_res50_9/checkpoint_epoch17_batch180000.pt':
            classifier = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
            classifier.to(device)
            print(f"Loading checkpoint from {classifier_path}...")
            checkpoint = torch.load(classifier_path, map_location=device)
            classifier.load_state_dict(checkpoint['model_state_dict'])
            classifier.eval()
        
            weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
            preprocess_valid = weights.transforms()
            sampler_kwargs['classifier_preprocess'] = preprocess_valid
        else:
            raise KeyError(f'The map of this classifier is not implemented:{classifier_path}')
        
        classifier.to(device).eval()
    # Create Sampler
    if sampler not in guidance_config:
        raise KeyError(f"Unknown sampler: {sampler}")
    
    # Add some arguments into sampler_kwargs to pass into sampler
    sampler_kwargs.setdefault('guidance_scheduler', guidance_scheduler)
    sampler_kwargs.setdefault('debug', debug)
    
    sampler = guidance_config[sampler](model, encoder=vae,classifier=classifier, pca=pca, **sampler_kwargs)
    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()
        
    

    # Divide seeds into batches.
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    if verbose:
        dist.print0(f'Generating {len(seeds)} images...')
    save_classifier_stats = sampler_kwargs.get('save_classifier_stats',False)
    dist.print0(f'save_classifier_stats:{save_classifier_stats}')
    # Return an iterable over the batches.
    class ImageIterable:
        def __len__(self):
            return len(rank_batches)

        def __iter__(self):
            # Loop over batches.
            assert outdir is not None
            all_classifier_results = []
            all_trajectories = []
            trajectory_labels = []
            with torch.no_grad():
                for batch_idx, indices in enumerate(rank_batches):
                    batch = dnnlib.EasyDict(images=None, labels=None, noise=None, batch_idx=batch_idx, num_batches=len(rank_batches), indices=indices)
                    batch.seeds = [seeds[idx] for idx in indices]
                    if len(batch.seeds) > 0:
    
                        # Pick noise and labels.
                        rnd = StackedRandomGenerator(device, batch.seeds)
                        batch.noise = rnd.randn([len(batch.seeds), 4, 64, 64], device=device)
                        ###檢查項目
                        # z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
                        # y = torch.randint(0, args.num_classes, (n,), device=device)
                        ###檢查項目
                        batch.labels = torch.eye(1000, device=device)[rnd.randint(1000, size=[len(batch.seeds)], device=device)]
                        batch.labels = torch.argmax(batch.labels,axis=1)
                        debug_print(debug, "noise:", batch.noise.min().item(), batch.noise.max().item())
                        debug_print(debug, "noise is nan?", torch.isnan(batch.noise).any().item())
    
                        # Generate images.
                        # classifier_results = sampler.sample(noise=batch.noise, labels=batch.labels,seeds=batch.seeds)
                        
                        
                        samples = sampler.sample(noise=batch.noise, y=batch.labels,seeds=batch.seeds)

                        samples = vae.decode(samples / 0.18215).sample
                        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()


                        
                        # Save images.
                        if outdir is not None:
                            for seed, image in zip(batch.seeds, samples.permute(0, 2, 3, 1).cpu().numpy()):
                                image_dir = os.path.join(outdir, f'{seed//1000*1000:06d}') if subdirs else outdir
                                os.makedirs(image_dir, exist_ok=True)
                                PIL.Image.fromarray(samples, 'RGB').save(os.path.join(image_dir, f'{seed:06d}.png'))
                        # Yield results.
                    torch.distributed.barrier() # keep the ranks in sync
                    yield batch
                # ===== 所有 batch 處理後才畫出完整的 PCA trajectory 圖 =====
                if dist.get_rank() == 0 and outdir is not None and len(all_classifier_results)!=0:
                    os.makedirs(outdir, exist_ok=True)
                    all_classifier_results_tensor = torch.cat(all_classifier_results,axis=0)
                    debug_print(debug, f"all_classifier_results_tensor.shape:{all_classifier_results_tensor.shape}")
                    debug_print(debug, f"all_classifier_results_tensor:{all_classifier_results_tensor}")
                    
                    torch.save(all_classifier_results_tensor, os.path.join(outdir, f'classifier_results.pt'))
                

    return ImageIterable()
#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option('--net_path',                      help='Main network pickle filename', metavar='PATH|URL',                type=str, default=None)
@click.option('--classifier_path',          help='Classifier .pt filename', metavar='PATH|URL',                     type=str, default=None)

@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',                  type=str, required=True)
@click.option('--subdirs',                  help='Create subdirectory for every 1000 seeds',                        is_flag=True)
@click.option('--seeds',                    help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST',            type=parse_int_list, default='16-19', show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                               type=click.IntRange(min=1), default=32, show_default=True)

@click.option('--steps', 'num_steps',       help='Number of sampling steps', metavar='INT',                         type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--sigma_min',                help='Lowest noise level', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True)
@click.option('--sigma_max',                help='Highest noise level', metavar='FLOAT',                            type=click.FloatRange(min=0, min_open=True), default=80, show_default=True)
@click.option('--guidance',                 help='Guidance strength  [default: 1; no guidance]', metavar='FLOAT',   type=float, default=None)
@click.option('--sampler', 'sampler',       help='The sampler to run', metavar='STR',                         type=str, default='cfg', show_default=True)

@click.option('--guidance_scheduler', 'guidance_scheduler',       help='The scheduler of guidance scale', metavar='STR',                         type=str, default='const_scheduler', show_default=True)
@click.option('--debug', 'debug',       help='Flag of debug message', metavar='BOOL',                         type=bool, default=False, show_default=True)
@click.option('--heun_guid', 'heun_guid',help='Whether use guidance in huen approximation for any guidance sampler', metavar='BOOL',type=bool, default=False, show_default=True)

@click.option('--w_interval_low_sigma', 'w_interval_low_sigma',       help='left edge of w interval', metavar='FLOAT',                         type=float, default=0.28, show_default=True)
@click.option('--w_interval_high_sigma', 'w_interval_high_sigma',       help='right edge of w interval', metavar='FLOAT',                         type=float, default=2.9, show_default=True)
@click.option('--w_interval_low_steps', 'w_interval_low_steps',       help='left edge of w interval', metavar='INT',                         type=int, default=16, show_default=True)
@click.option('--w_interval_high_steps', 'w_interval_high_steps',       help='right edge of w interval', metavar='INT',                         type=int, default=27, show_default=True)
@click.option('--w_interval_low_peak_steps', 'w_interval_low_peak_steps',       help='left edge on peak of w interval in trapezoid scheduler', metavar='INT',                         type=int, default=20, show_default=True)
@click.option('--w_interval_high_peak_steps', 'w_interval_high_peak_steps',       help='right edge on peak of w interval in trapezoid scheduler', metavar='INT',                         type=int, default=25, show_default=True)
@click.option('--w_interval_high_middle_steps', 'w_interval_high_middle_steps',       help='right edge on peak of w interval in trapezoid scheduler', metavar='INT',                         type=int, default=25, show_default=True)
@click.option('--w_skewed',       help='ex:True', metavar='FLOAT',                         type=float, default=0.38, show_default=True)
@click.option('--image_size',       help='ex:64', metavar='INT',                         type=int)
@click.option('--save_classifier_stats',       help='ex:True', metavar='BOOL',                         type=bool, default=False, show_default=True)
@click.option('--adaptive_min_prob',       help='ex:True', metavar='FLOAT',                         type=float, default=0.4, show_default=True)
@click.option('--adaptive_max_prob',       help='ex:True', metavar='FLOAT',                         type=float, default=0.4, show_default=True)
@click.option('--adaptive_prob',       help='ex:True', metavar='FLOAT',                         type=float, default=0.4, show_default=True)
@click.option('--open_prob',       help='ex:True', metavar='FLOAT',                         type=float, default=0.6, show_default=True)
@click.option('--close_prob',       help='ex:True', metavar='FLOAT',                         type=float, default=0.4, show_default=True)
@click.option('--hold_steps',       help='ex:128', metavar='INT',                         type=int)
@click.option('--balance_coeff',       help='ex:True', metavar='FLOAT',                         type=float, default=3.0, show_default=True)
def cmdline(**opts):
    """Generate random images using the given model.

    Examples:

    \b
    # Generate a couple of images and save them as out/*.png
    python generate_images.py --preset=edm2-img512-s-guid-dino --outdir=out

    \b
    # Generate 50000 images using 8 GPUs and save them as out/*/*.png
    torchrun --standalone --nproc_per_node=8 generate_images.py \\
        --preset=edm2-img64-s-fid --outdir=out --subdirs --seeds=0-49999
    """
    opts = dnnlib.EasyDict(opts)

    if opts.sampler=='noguid_stats' or opts.sampler=='cfg_stats':
        if opts.classifier_path is None:
            raise click.ClickException('Please specify --classifier_path when using sampler that required classifier')
    

    # Generate.
    dist.init()
    image_iter = generate_images_complete(**opts)
    for _r in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
