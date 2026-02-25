from .UNET import UNet
from .RadioUNET import RadioWNet
from .pmnet_v3 import PMNet
from .REM_NET import REM_Net
from .AE import Autoencoder
# from .LUANet_v2 import LUANet_v2_test
from .PathFinder import PathFinder

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
    elif model_name == 'UNet_gate' or model_name == 'UGNet':
        model = UNet_gate(**model_params)
    elif model_name == 'UNet_attn':
        model = UNet_attn(**model_params)
    elif model_name == 'UNet_attn_d':
        model = UNet_attn_d(**model_params)
    elif model_name == 'AE':
        model = Autoencoder()
    elif model_name == 'LUANet':
        model = LUANet(**model_params)
        # model = LUANet_v2_test(**model_params)
    elif model_name == 'VIT':
        model = ViT()
    elif model_name == 'PathFinder':
        model = PathFinder()
    return model