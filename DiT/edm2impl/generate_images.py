import os
import torch
import numpy as np
import joblib
import PIL.Image
import click
import re

from base.generate_images import config_presets
import base.dnnlib as dnnlib
from base.generate_images import StackedRandomGenerator
from base.torch_utils import distributed as dist

from sampler.guidance import guidance_config
from plot_trajectories import plot_all_trajectories
from utils import debug_print
from edm2impl.utils import load_network
import tqdm

def generate_images_complete(
    net,                                        # Main network.
    gnet,                                       # Guiding network
    encoder             = None,                 # Instance of training.encoders.Encoder. None = load from network pickle.
    outdir              = None,                 # Where to save the output images. None = do not save.
    subdirs             = False,                # Create subdirectory for every 1000 seeds?
    seeds               = range(16, 24),        # List of random seeds.
    class_idx           = None,                 # Class label. None = select randomly.
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
    ##############################################################
    sampler             = 'cfg',                # Name of the sampler.
    **sampler_kwargs,                           # Additional arguments for the sampler function.
):
    # Load PCA model
    pca = None
    if pca_plot:
        dist.print0(f'Plot PCA in this generation.')
        try:
            pca = joblib.load(pca_path)
            dist.print0(f'Complete PCA loading.')
        except:
            raise Exception(f'Failed to load pca at path:{pca_path} while pca_plot flag is true')
    # Create Sampler
    if sampler not in guidance_config:
        raise KeyError(f"Unknown sampler: {sampler}")
    
    # Add some arguments into sampler_kwargs to pass into sampler
    sampler_kwargs.setdefault('guidance_scheduler', guidance_scheduler)
    sampler_kwargs.setdefault('debug', debug)
    sampler = guidance_config[sampler](net, gnet=gnet, encoder=encoder, pca=pca, **sampler_kwargs)
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()
        
    # Load PCA model
    pca = None
    if pca_plot:
        dist.print0(f'Plot PCA in this generation.')
        try:
            pca = joblib.load(pca_path)
            dist.print0(f'Complete PCA loading.')
        except:
            raise Exception(f'Failed to load pca at path:{pca_path} while pca_plot flag is true')

    # Initialize encoder.
    assert encoder is not None
    if verbose:
        dist.print0(f'Setting up {type(encoder).__name__}...')
    encoder.init(device)
    if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
        encoder.batch_size = encoder_batch_size

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide seeds into batches.
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    if verbose:
        dist.print0(f'Generating {len(seeds)} images...')

    # Return an iterable over the batches.
    class ImageIterable:
        def __len__(self):
            return len(rank_batches)

        def __iter__(self):
            # Loop over batches.
            all_trajectories = []  # list of [T, 2]
            trajectory_labels = []
            for batch_idx, indices in enumerate(rank_batches):
                batch = dnnlib.EasyDict(images=None, labels=None, noise=None, batch_idx=batch_idx, num_batches=len(rank_batches), indices=indices)
                batch.seeds = [seeds[idx] for idx in indices]
                if len(batch.seeds) > 0:

                    # Pick noise and labels.
                    rnd = StackedRandomGenerator(device, batch.seeds)
                    batch.noise = rnd.randn([len(batch.seeds), net.img_channels, net.img_resolution, net.img_resolution], device=device)
                    if net.label_dim > 0:
                        batch.labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[len(batch.seeds)], device=device)]
                        if class_idx is not None:
                            batch.labels[:, :] = 0
                            batch.labels[:, class_idx] = 1
                    debug_print(debug, "noise:", batch.noise.min().item(), batch.noise.max().item())
                    debug_print(debug, "noise is nan?", torch.isnan(batch.noise).any().item())

                    # Generate images.
                    latents, pca_features = sampler.sample(noise=batch.noise, labels=batch.labels,seeds=batch.seeds)
                    all_trajectories.append(pca_features)
                    label_indices = torch.argmax(batch.labels, dim=1).tolist()
                    trajectory_labels.extend(label_indices)
                    debug_print(debug, "latents:", latents.min().item(), latents.max().item())
                    batch.images = encoder.decode(latents)
                    # Save images.
                    if outdir is not None:
                        for seed, image in zip(batch.seeds, batch.images.permute(0, 2, 3, 1).cpu().numpy()):
                            image_dir = os.path.join(outdir, f'{seed//1000*1000:06d}') if subdirs else outdir
                            os.makedirs(image_dir, exist_ok=True)
                            PIL.Image.fromarray(image, 'RGB').save(os.path.join(image_dir, f'{seed:06d}.png'))
                # Yield results.
                torch.distributed.barrier() # keep the ranks in sync
                yield batch
            # ===== 所有 batch 處理後才畫出完整的 PCA trajectory 圖 =====
            if dist.get_rank() == 0 and pca_plot and len(all_trajectories) > 0:
                plot_all_trajectories(
                    trajectories= np.concatenate(all_trajectories, axis=0),
                    trajectory_labels=trajectory_labels,
                    out_path=os.path.join(outdir, "all_pca_trajectories.png"), # type: ignore
                    debug=debug,
                )

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
@click.option('--preset',                   help='Configuration preset', metavar='STR',                             type=str, default=None)
@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',                  type=str, required=True)
@click.option('--subdirs',                  help='Create subdirectory for every 1000 seeds',                        is_flag=True)
@click.option('--seeds',                    help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST',            type=parse_int_list, default='16-19', show_default=True)
@click.option('--class', 'class_idx',       help='Class label  [default: random]', metavar='INT',                   type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                               type=click.IntRange(min=1), default=1, show_default=True)

@click.option('--steps', 'num_steps',       help='Number of sampling steps', metavar='INT',                         type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--sigma_min',                help='Lowest noise level', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True)
@click.option('--sigma_max',                help='Highest noise level', metavar='FLOAT',                            type=click.FloatRange(min=0, min_open=True), default=80, show_default=True)
@click.option('--rho',                      help='Time step exponent', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--guidance',                 help='Guidance strength  [default: 1; no guidance]', metavar='FLOAT',   type=float, default=None)
@click.option('--S_churn', 'S_churn',       help='Stochasticity strength', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',           help='Stoch. min noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',           help='Stoch. max noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',       help='Stoch. noise inflation', metavar='FLOAT',                         type=float, default=1, show_default=True)
@click.option('--sampler', 'sampler',       help='The sampler to run', metavar='STR',                         type=str, default='cfg', show_default=True)
@click.option('--guidance_scheduler', 'guidance_scheduler',       help='The scheduler of guidance scale', metavar='STR',                         type=str, default='const_scheduler', show_default=True)
@click.option('--debug', 'debug',       help='Flag of debug message', metavar='BOOL',                         type=bool, default=False, show_default=True)
@click.option('--pca_plot', 'pca_plot',       help='Flag of plot pca trajactory', metavar='BOOL',                         type=bool, default=False, show_default=True)

def cmdline(preset, **opts):
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

    # Apply preset.
    if preset is not None:
        if preset not in config_presets:
            raise click.ClickException(f'Invalid configuration preset "{preset}"')
        net, gnet, encoder = load_network(config_presets[preset].net, config_presets[preset].gnet)
        opts.net = net
        opts.gnet = encoder
        opts.encoder = encoder

    # Generate.
    dist.init()
    image_iter = generate_images_complete(**opts)
    for _r in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
