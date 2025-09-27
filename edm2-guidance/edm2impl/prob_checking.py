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
from sampler.guidances.ej_ddg import ej_forward_constructor
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
    classifier_path     = None,                 # Path of classifier model .pt file
    ##############################################################
    sampler             = 'cfg',                # Name of the sampler.
    **sampler_kwargs,                           # Additional arguments for the sampler function.
):
    
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    # Load main network.
    dist.print0(f'Checking net:{net}')
    if isinstance(net, str):
        if verbose:
            dist.print0(f'Loading main network from {net} ...')
        with dnnlib.util.open_url(net, verbose=(verbose and dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        net = data['ema'].to(device)
        if encoder is None:
            encoder = data.get('encoder', None)
            if encoder is None:
                encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')
    assert net is not None

    # Load guidance network.
    if isinstance(gnet, str):
        if verbose:
            dist.print0(f'Loading guiding network from {gnet} ...')
        if sampler =='ej_ddg':
            # gent = net.copy()
            gnet = copy.deepcopy(net)
            ej_forward = ej_forward_constructor(
                debug=debug,theta=sampler_kwargs.get('ej_theta'),
                 z_normalize=sampler_kwargs.get('ej_z_normalize'),
                 rho=sampler_kwargs.get('ej_rho'),
                 rho_noise=sampler_kwargs.get('ej_rho_noise'),)
            gnet.unet.forward = types.MethodType(ej_forward, gnet.unet)
            dist.print0(f'Replace gent by net and replace the forward function of net by ej_ddg')
        else:
            with dnnlib.util.open_url(gnet, verbose=(verbose and dist.get_rank() == 0)) as f:
                gnet = pickle.load(f)['ema'].to(device)
            
    if gnet is None:
        gnet = net

    # Initialize encoder.
    assert encoder is not None
    if verbose:
        dist.print0(f'Setting up {type(encoder).__name__}...')
    encoder.init(device)
    if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
        encoder.batch_size = encoder_batch_size
    # Load PCA model
    pca = None
    if pca_plot:
        dist.print0(f'Plot PCA in this generation.')
        try:
            pca = joblib.load(pca_path)
            dist.print0(f'Complete PCA loading.')
        except:
            raise Exception(f'Failed to load pca at path:{pca_path} while pca_plot flag is true')
    # Load classifier(if using sampler related with classifier)
    classifier=None
    if sampler=='map_neg_ddg' or sampler=='cg' or sampler=='map_x0pred':
        assert classifier_path is not None
    if classifier_path is not None:
        # classifier_kwargs = {k: v for k, v in sampler_kwargs.items() if (k.startswith('classifier_') or k.startswith('image_size')) and v is not None}
        # # print('classifier_kwargs:',classifier_kwargs)
        # classifier_args = classifier_defaults()
        # # print('classifier_default_args:',classifier_args)
        # classifier_args.update(classifier_kwargs)
        # # print('classifier_args:',classifier_args)
        # classifier = create_classifier(
        #     **classifier_args
        # ).to(device)
        # state_dict = torch.load(classifier_path, map_location=device)
        # classifier.load_state_dict(state_dict)
        num_classes = 1000
        in_channels = 4
        ###### res50 classifier
        # classifier = create_resnet50_classifier(num_classes, in_channels, False)
        # classifier.to(device)
        # print(f"Loading classifier from {classifier_path}...")
        # checkpoint = torch.load(classifier_path, map_location=device)
        # classifier.load_state_dict(checkpoint['model_state_dict'])
        ###### res 50 classifier
        ###### cl_10
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
    assert sampler =='map_stats' or sampler =='noguid_stats' or sampler == 'cfg_stats' or sampler == 'cfg_adaptive' or sampler == 'cfg_adaptive_general' or sampler == 'map_adaptive' or sampler == 'cfg_adaptive_v9' or sampler == 'cfg_adaptive_v10' or sampler == 'map_adaptive_v10' or sampler == 'map_ddg_x0pred_checking' or sampler == 'cfg_adaptive_v11'
    sampler = guidance_config[sampler](net, gnet=gnet, encoder=encoder,classifier=classifier, pca=pca, **sampler_kwargs)
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
                        batch.noise = rnd.randn([len(batch.seeds), net.img_channels, net.img_resolution, net.img_resolution], device=device)
                        if net.label_dim > 0:
                            batch.labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[len(batch.seeds)], device=device)]
                            if class_idx is not None:
                                batch.labels[:, :] = 0
                                batch.labels[:, class_idx] = 1
                        debug_print(debug, "noise:", batch.noise.min().item(), batch.noise.max().item())
                        debug_print(debug, "noise is nan?", torch.isnan(batch.noise).any().item())
    
                        # Generate images.
                        # classifier_results = sampler.sample(noise=batch.noise, labels=batch.labels,seeds=batch.seeds)
                        
                        
                        latents, pca_features,classifier_results = sampler.sample(noise=batch.noise, labels=batch.labels,seeds=batch.seeds)
                        classifier_results = classifier_results.to('cpu')
                        if save_classifier_stats:
                            all_classifier_results.append(classifier_results)
                        debug_print(debug, f"classifier_results.shape:{classifier_results.shape:}")
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
@click.option('--preset',                   help='Configuration preset', metavar='STR',                             type=str, default=None)
@click.option('--net',                      help='Main network pickle filename', metavar='PATH|URL',                type=str, default=None)
@click.option('--gnet',                     help='Guiding network pickle filename', metavar='PATH|URL',             type=str, default=None)
@click.option('--classifier_path',          help='Classifier .pt filename', metavar='PATH|URL',                     type=str, default=None)

@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',                  type=str, required=True)
@click.option('--subdirs',                  help='Create subdirectory for every 1000 seeds',                        is_flag=True)
@click.option('--seeds',                    help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST',            type=parse_int_list, default='16-19', show_default=True)
@click.option('--class', 'class_idx',       help='Class label  [default: random]', metavar='INT',                   type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                               type=click.IntRange(min=1), default=32, show_default=True)

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
@click.option('--delta', 'delta',       help='delta for time offset sampling', metavar='FLOAT',                         type=float, default=False, show_default=True)
@click.option('--delta', 'delta',       help='delta for time offset sampling', metavar='FLOAT',                         type=float, default=0.01, show_default=True)
@click.option('--k_average', 'k_average',       help='k for random ddg first k negative labels', metavar='INT',                         type=int, default=1, show_default=True)
@click.option('--negative_class_csv_path', 'negative_class_csv_path',       help='Path for negative class csv', metavar='STR',                         type=str, default='./plot/negative_class_manual.csv', show_default=True)
@click.option('--full_random', 'full_random',       help='Whether random 1000 classes for random ddg', metavar='BOOL',                         type=bool, default=False, show_default=True)
@click.option('--n_random', 'n_random',       help='random choose n scores and take mean for random ddg', metavar='INT',                         type=int, default=1000, show_default=True)
@click.option('--weighted_random',       help='Whether use information of given negative label csv to weighted the random choice in random ddg', metavar='BOOL',                         type=bool, default=False, show_default=True)
@click.option('--heun_guid', 'heun_guid',       help='Whether use guidance in huen approximation for any guidance sampler', metavar='BOOL',                         type=bool, default=False, show_default=True)
@click.option('--delta_scheduler', 'delta_scheduler',       help='What delta scheduler used in TOG delta function', metavar='STR',                         type=str, default='linear_decrease_scheduler', show_default=True)
@click.option('--use_gnet', 'use_gnet',       help='Whether use gnet in TOG', metavar='BOOL',                         type=bool, default=False, show_default=True)
@click.option('--w_interval_low_sigma', 'w_interval_low_sigma',       help='left edge of w interval', metavar='FLOAT',                         type=float, default=0.28, show_default=True)
@click.option('--w_interval_high_sigma', 'w_interval_high_sigma',       help='right edge of w interval', metavar='FLOAT',                         type=float, default=2.9, show_default=True)
@click.option('--w_interval_low_steps', 'w_interval_low_steps',       help='left edge of w interval', metavar='INT',                         type=int, default=16, show_default=True)
@click.option('--w_interval_high_steps', 'w_interval_high_steps',       help='right edge of w interval', metavar='INT',                         type=int, default=27, show_default=True)
@click.option('--w_interval_low_peak_steps', 'w_interval_low_peak_steps',       help='left edge on peak of w interval in trapezoid scheduler', metavar='INT',                         type=int, default=20, show_default=True)
@click.option('--w_interval_high_peak_steps', 'w_interval_high_peak_steps',       help='right edge on peak of w interval in trapezoid scheduler', metavar='INT',                         type=int, default=25, show_default=True)
@click.option('--w_interval_high_middle_steps', 'w_interval_high_middle_steps',       help='right edge on peak of w interval in trapezoid scheduler', metavar='INT',                         type=int, default=25, show_default=True)
@click.option('--w_skewed',       help='ex:True', metavar='FLOAT',                         type=float, default=0.38, show_default=True)
@click.option('--image_size',       help='ex:64', metavar='INT',                         type=int)
@click.option('--classifier_attention_resolutions',       help='ex:32,16,8', metavar='STR',                         type=str)
@click.option('--classifier_depth',       help='ex:4', metavar='INT',                         type=int)
@click.option('--classifier_width',       help='ex:128', metavar='INT',                         type=int)
@click.option('--classifier_pool',       help='ex:attention', metavar='STR',                         type=str)
@click.option('--classifier_resblock_updown',       help='ex:True', metavar='BOOL',                         type=bool)
@click.option('--classifier_use_scale_shift_norm',       help='ex:True', metavar='BOOL',                         type=bool)
@click.option('--ej_z_normalize',       help='ex:True', metavar='BOOL',                         type=bool, default=True, show_default=True)
@click.option('--ej_theta',       help='ex:True', metavar='FLOAT',                         type=float, default=20.0, show_default=True)
@click.option('--ej_rho_noise',       help='ex:True', metavar='BOOL',                         type=bool, default=False, show_default=True)
@click.option('--ej_rho',       help='ex:True', metavar='FLOAT',                         type=float, default=1.0, show_default=True)
@click.option('--avoid_self_label',       help='ex:True', metavar='BOOL',                         type=bool, default=True, show_default=True)
@click.option('--save_classifier_stats',       help='ex:True', metavar='BOOL',                         type=bool, default=False, show_default=True)
@click.option('--adaptive_min_prob',       help='ex:True', metavar='FLOAT',                         type=float, default=0.4, show_default=True)
@click.option('--adaptive_max_prob',       help='ex:True', metavar='FLOAT',                         type=float, default=0.4, show_default=True)
@click.option('--adaptive_prob',       help='ex:True', metavar='FLOAT',                         type=float, default=0.4, show_default=True)
@click.option('--open_prob',       help='ex:True', metavar='FLOAT',                         type=float, default=0.6, show_default=True)
@click.option('--close_prob',       help='ex:True', metavar='FLOAT',                         type=float, default=0.4, show_default=True)
@click.option('--hold_steps',       help='ex:128', metavar='INT',                         type=int)
@click.option('--balance_coeff',       help='ex:True', metavar='FLOAT',                         type=float, default=3.0, show_default=True)
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
        for key, value in config_presets[preset].items():
            if opts[key] is None:
                opts[key] = value

    # Validate options.
    if opts.net is None:
        raise click.ClickException('Please specify either --preset or --net')
    if opts.guidance is None or opts.guidance == 1:
        opts.guidance = 1
        opts.gnet = None
    elif opts.gnet is None:
        raise click.ClickException('Please specify --gnet when using guidance')
    if opts.sampler=='map_neg_ddg' or opts.sampler=='cg':
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
