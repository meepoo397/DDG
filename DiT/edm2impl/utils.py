import pickle
from typing import Iterable, Sized
import torch
from base import dnnlib
from base.torch_utils import distributed as dist

def debug_print(debug=False,*message_args):
    if debug:
        dist.print0('[DEBUG] ',*message_args)

def load_network(net, gnet=None, encoder=None, verbose=True, device=torch.device('cuda')):
    if isinstance(net, str):
        if verbose:
            dist.print0(f'Loading main network from {net} ...')
        with dnnlib.util.open_url(net, verbose=(verbose and dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        net = data['ema'].to(device)
        if encoder is None:
            encoder = data.get('encoder', None)
            if encoder is None:
                encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder').to(device)
    assert net is not None

    # Load guidance network.
    if isinstance(gnet, str):
        if verbose:
            dist.print0(f'Loading guiding network from {gnet} ...')
        with dnnlib.util.open_url(gnet, verbose=(verbose and dist.get_rank() == 0)) as f:
            gnet = pickle.load(f)['ema'].to(device)
    if gnet is None:
        gnet = net
    
    return net, gnet, encoder

def load_classifier(path = 'classifier/512x512_classifier.pt'):
    from classifier.script_util import create_classifier, classifier_defaults

    dist.print0("loading classifier...")
    d = classifier_defaults()
    d['image_size'] = 512
    d['classifier_depth'] = 2
    classifier = create_classifier(**d)
    classifier.load_state_dict(
        torch.load(path, map_location=torch.device("cuda"), weights_only=True)
    )
    classifier.to(torch.device('cuda'))
    return classifier.eval()
