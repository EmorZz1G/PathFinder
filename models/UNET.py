import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.GroupNorm(8,in_channels) if in_channels != 2 else nn.Identity(),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8,out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        skip = self.skip(x)
        x = self.double_conv(x) + skip
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            # nn.MaxPool2d(2),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1),
            DoubleConv(in_channels, out_channels),
            DoubleConv(out_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2)
            self.conv = nn.Sequential(DoubleConv(in_channels, out_channels),DoubleConv(out_channels, out_channels))
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = nn.Sequential(DoubleConv(in_channels, out_channels),DoubleConv(out_channels, out_channels))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=2, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        c_dims = [64 * 2 ** i for i in range(6)]

        self.inc = DoubleConv(n_channels, c_dims[0])
        self.down1 = Down(c_dims[0], c_dims[1])
        self.down2 = Down(c_dims[1], c_dims[2])
        self.down3 = Down(c_dims[2], c_dims[3])
        self.down4 = Down(c_dims[3], c_dims[4])
        
        self.bottleneck = DoubleConv(c_dims[4], c_dims[3])
        
        self.up1 = Up(c_dims[4], c_dims[2])
        self.up2 = Up(c_dims[3], c_dims[1])
        self.up3 = Up(c_dims[2], c_dims[0])
        self.up4 = Up(c_dims[1], c_dims[1])
        self.outc = nn.Sequential(
                         nn.Conv2d(c_dims[1]+2, 64, kernel_size=3, padding=1),
                         nn.GroupNorm(8,64),
                         nn.ReLU(),
                         nn.Conv2d(64, 64, kernel_size=3, padding=1),
                         nn.GroupNorm(8, 64),
                         nn.ReLU(),
                         nn.Conv2d(64, 1, kernel_size=3, padding=1))

    def forward(self, x0):
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x5 = self.bottleneck(x5)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # x (bs, 128, 256, 256)
        
        x = torch.cat([x, x0], dim=1)
        
        x = self.outc(x)
        return x
    
import yaml

if __name__ == '__main__':
    tmp = __file__
    import os
    tmp = os.path.dirname(tmp)
    print(tmp)
    import torch
    def load_model_from_yaml(config):
        with open(config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        model_name = config['model']['name']
        model_params = config['model']['params']
        if model_name == 'UNet':
            model = UNet(**model_params)
        return model
    
    # 使用示例
    # model = load_model_from_yaml(tmp + '/../config/default.yaml')
    
    model = UNet()
    # 可以进一步进行模型相关操作，比如放到GPU等
    def cal_model_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)/1024/1024
    
    print(cal_model_params(model))
    # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    # bs = 64
    # x = torch.randn(bs, 2, 256, 256)
    
    # from time import time
    
    # st = time()
    # for _ in range(100):
    #     output = model(x)
    #     tm = time()-st
    #     # print('Time:', tm)
        
    # print("total time:", time()-st)
            # print(output.shape)
        # output = model(x)
        # print(output.shape)