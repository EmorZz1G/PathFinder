from .UNET import UNet
from .RadioUNET import RadioWNet
from .pmnet_v3 import PMNet
from .REM_NET import REM_Net
from .AE import Autoencoder
from .PathFinder import PathFinder
from .PathFinder_wo_low_rank import PathFinder_wo_low_rank

def load_model_from_yaml(config):
    model_name = config['model']['name']
    model_params = config['model']['params']
    if model_name == 'UNet':
        model = UNet(**model_params)
    elif model_name == 'RadioUNet':
        model = RadioWNet(**model_params)
    elif model_name == 'PMNet':
        model = PMNet()
    elif model_name == 'REM_Net':
        model = REM_Net()
    elif model_name == 'AE':
        model = Autoencoder()
    elif model_name == 'PathFinder':
        model = PathFinder()
    elif model_name == 'PathFinder_wo_low_rank':
        model = PathFinder_wo_low_rank()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model