import torch
import torch.nn as nn
class DoubleConv(nn.Module):
    """(convolution => [BN] => SiLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.GroupNorm(32, in_channels) if in_channels != 2 else nn.Identity(),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        )
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        skip = self.skip(x)
        x = self.double_conv(x) + skip
        return x
    
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, height=256,
                 width=256,
                 c_len=128,
                 n_filters=256, activ_function_name=None, est_separately_accr_freq=False, kernel_size=(3, 3), conv_stride=1, pool_size_str=2, use_batch_norm=True):
        super(Autoencoder, self).__init__()
        self.height = height
        self.width = width
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.pool_size_str = pool_size_str
        self.use_batch_norm = use_batch_norm
        self.c_len = c_len
        self.norm_hereToo = False
        self.est_separately_accr_freq = est_separately_accr_freq
        self.activ_func = getattr(F, activ_function_name) if activ_function_name else F.leaky_relu

        n_channels = 2

        # Encoder
        self.encoder = nn.Sequential(
            DoubleConv(n_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            DoubleConv(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            DoubleConv(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            DoubleConv(512, 512, kernel_size=3, stride=1, padding=1),
        )

    
        self.decoder = nn.Sequential(
            DoubleConv(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DoubleConv(512, 256, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DoubleConv(256, 128, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        # x = self.flatten(x)
        # x = self.latent_dense(x)
        # x = self.decoder_input(x)
        # x = x.view(x.size(0), self.n_filters, self.height // (self.pool_size_str ** 2), self.width // (self.pool_size_str ** 2))
        x = self.decoder(x).contiguous()
        # x = 10 * torch.log10(x)
        # x = x.view(x.size(0), -1, 1)
        return x
    
    
if __name__ == '__main__':
    model = Autoencoder()
    print(model)
    x = torch.randn(1, 2, 256, 256)
    print(model(x).shape)