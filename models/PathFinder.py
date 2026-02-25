import torch
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='same', dilation=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True),
            nn.GroupNorm(16,out_channels) if out_channels >3 else nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=True),
            nn.GroupNorm(16,out_channels) if out_channels >3 else nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=True)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        # print('1',x.shape)
        skip = self.skip(x)
        x = self.double_conv(x) + skip
        # print('2',x.shape)
        return x



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class AE(torch.nn.Module):
    def __init__(self, in_channel, dims=[256,128], strides=[1, 1, 1, 1], ds=[1, 1, 1, 1]):
        super(AE, self).__init__()
        
        self.encoder = []
        self.decoder = []
        
        for i in range(len(dims)):
            if i==0:
                self.encoder.append(DoubleConv(in_channel,dims[i],dilation=ds[i]))
                self.decoder.append(DoubleConv(dims[i],in_channel,dilation=ds[i]))
            elif i>0:
                self.encoder.append(DoubleConv(dims[i-1],dims[i],dilation=ds[i]))
                self.decoder.append(DoubleConv(dims[i],dims[i-1],dilation=ds[i]))
                
            if strides[i] == 2:
                self.encoder.append(nn.Conv2d(dims[i], dims[i], kernel_size=3, stride=2, padding=1))
                # self.decoder.append(nn.ConvTranspose2d(dims[i], dims[i], kernel_size=3, stride=2, padding=1, output_padding=1))
                self.decoder.append(nn.Conv2d(dims[i], dims[i], kernel_size=3, stride=1, padding=1))
                self.decoder.append(nn.Upsample(scale_factor=2))
        
        
        self.decoder = self.decoder[::-1]
        
        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x

try:    
    # from .UNET import UNet2, UNets
    from .attention import BGAttention, BGAttentionReduce
except:
    # from UNET import UNet2, UNets
    from attention import BGAttention, BGAttentionReduce


def find_and_pad_points(images):
    images = images.squeeze(1)
    # 找到所有大于 0 的点的索引
    indices = torch.nonzero(images > 0)
    # 初始化结果张量，形状为 (batch_size, 2, 2)，用于存储最多 2 个点的坐标
    result = -torch.ones((images.shape[0], 2, 2), dtype=torch.long).to(images.device)
    # 对于每个样本，将找到的点存储到结果张量中
    batch_indices = indices[:, 0]
    valid_indices = indices[:, 1:]
    for i in range(images.shape[0]):
        # 找到当前批次中大于 0 的点
        batch_valid_indices = valid_indices[batch_indices == i]
        num_points = min(batch_valid_indices.shape[0], 2)
        # 将找到的点存储到结果张量中
        result[i, :num_points] = batch_valid_indices[:num_points]
    return result

def get_prompt(vecs, result):
    ret_vecs = torch.zeros((vecs.shape[0], 2, vecs.shape[1]), dtype=vecs.dtype).to(vecs.device)
    for batch_i, x in enumerate(result):
        for val_i, y in enumerate(x):
            i = y[0]
            j = y[1]
            if i == -1:
                continue
            ret_vecs[batch_i, val_i] = vecs[batch_i, :, i, j]
    return ret_vecs

    
class SwitchSequential(nn.Sequential):
    def forward(self, x, emb, bs_mask):
        for layer in self:
            if isinstance(layer, BGAttentionReduce):
                x = layer(x, emb, bs_mask)
            # elif isinstance(layer, GainAttention):
            #     x = layer(x, emb, ~bs_mask)
            else:
                x = layer(x)
        return x

    

class UNetBG(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, unet_layer = 4):
        super(UNetBG, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        c_dims = [64 * 2 ** i for i in range(6)]
        # (64, 128, 256, 512, 1024)
        
        tx_dim = n_channels

        self.inc = DoubleConv(n_channels, c_dims[0], kernel_size=1, stride=1, padding='same')
        
        self.encoders = []
        self.decoders = []
        # self.bottleneck = []
        
        enc_dims = []
        
        prev_dim = c_dims[0]
        for i in range(unet_layer):
            tmp = SwitchSequential(
                                   DoubleConv(prev_dim, c_dims[i]),
                                    DoubleConv(c_dims[i], c_dims[i]),
                                   BGAttentionReduce(c_dims[i], tx_dim)
                                #    Tx - Buliding
                                #   tx - Non-Building (20000 * 1)
                                   )
            self.encoders.append(tmp)
            enc_dims.append(c_dims[i])
            
            if i < unet_layer-1:
                self.encoders.append(SwitchSequential(nn.Conv2d(c_dims[i], c_dims[i], kernel_size=3, stride=2, padding=1)))
                enc_dims.append(c_dims[i])
            prev_dim = c_dims[i]
            
            
        self.bottleneck = SwitchSequential(DoubleConv(prev_dim, prev_dim),DoubleConv(prev_dim, prev_dim))
        
        last_dim = prev_dim
        for i in reversed(range(unet_layer)):
            prev_dim = enc_dims.pop()
            tmp = SwitchSequential(
                                      DoubleConv(last_dim+prev_dim, c_dims[i]),
                                      DoubleConv(c_dims[i], c_dims[i]),
                                      BGAttentionReduce(c_dims[i], tx_dim)
                                   )
            self.decoders.append(tmp)
            last_dim = c_dims[i]
            
            if i > 0:
                prev_dim = enc_dims.pop()
                tmp = SwitchSequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(last_dim+prev_dim, c_dims[i], kernel_size=3, stride=1, padding=1),
                )
                self.decoders.append(tmp)
                last_dim = c_dims[i]
            
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        
        self.outc = DoubleConv(last_dim, n_classes, kernel_size=1, stride=1, padding='same')

    def forward(self, x0, tx_emb, mask):
        skip_connections = []
        
        x = x0
        x = self.inc(x)
        for layers in self.encoders:
            # print(1, x.shape)
            x = layers(x, tx_emb, mask)
            # print(2,x.shape)
            skip_connections.append(x)

        x = self.bottleneck(x, tx_emb, mask)

        for layers in self.decoders:
            # print(3,x.shape)
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1)
            # print('cat',x.shape)
            x = layers(x, tx_emb, mask)
        
        x = self.outc(x)
        
        return x
    

class PathFinder(torch.nn.Module):
    
    def __init__(self, enc_dec_dims=[128,128]):
        super(PathFinder, self).__init__()
        
        in_channels = 1
        
        self.build_enc = []
        self.gain_dec = []
        
        for i in range(len(enc_dec_dims)):
            if i==0:
                self.build_enc.append(DoubleConv(in_channels, enc_dec_dims[i]))
                self.gain_dec.append(DoubleConv(enc_dec_dims[i], in_channels))
            elif i>0:
                self.build_enc.append(DoubleConv(enc_dec_dims[i-1], enc_dec_dims[i]))
                self.gain_dec.append(DoubleConv(enc_dec_dims[i], enc_dec_dims[i-1]))
            
        self.build_enc = nn.Sequential(*self.build_enc)
        self.gain_dec = self.gain_dec[::-1]
        self.gain_dec = nn.Sequential(*self.gain_dec)
        
        self.tx_emb = nn.Sequential(
            nn.Conv2d(in_channels, enc_dec_dims[-1], kernel_size=1, stride=1, padding='same'),
            nn.BatchNorm2d(enc_dec_dims[-1]),
            nn.ReLU(inplace=False),
            nn.Conv2d(enc_dec_dims[-1], enc_dec_dims[-1], kernel_size=1, stride=1, padding='same'),
        )
        
        self.unet = UNetBG(enc_dec_dims[-1], enc_dec_dims[-1])
        

    def forward(self, x):
        build, tx0 = x.chunk(2, dim=1)
        # x -> 建筑物，发射方
        mask = build > 0
        # 建筑
        build = self.build_enc(build)
        # 对建筑物编码
        tx_idx = find_and_pad_points(tx0)
        
        # 发射方，提示
        tx = self.tx_emb(tx0)
        # 对发射方编码
        
        tx_emb = get_prompt(tx, tx_idx)
        # print('tx_emb', tx_emb.shape, tx_emb)  
        # (Batch , Num_points, 128)
        
        # build + tx
        # attention
        x = self.unet(build + tx, tx_emb, mask)
        x = self.gain_dec(x)
        return x
    
    

def wrapper_forward_hook(model, data_input):
    import math
    
    import torch.nn.functional as F
    outputs = []
    
    def forward_hook(module, input, output):
        print('Hooked')
        x1, x2, y, mask = input
        # x (latent): # (Batch_Size, Seq_Len_Q, Dim_Q)
        # y (context): # (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 2, 768)
        input_shape = x1.shape
        batch_size, sequence_length, d_embed = input_shape
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, -1, module.n_heads, module.d_head)
    
        
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        q = module.q_proj(x1)
        q2 = module.q2_proj(x2)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        k = module.k_proj(y)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        v = module.v_proj(y)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = q.reshape(interim_shape).transpose(1, 2) 
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        k = k.reshape(interim_shape).transpose(1, 2) 
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        v = v.reshape(interim_shape).transpose(1, 2) 
        q2 = q2.reshape(interim_shape).transpose(1, 2)
        
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = q @ k.transpose(-1, -2)
        weight2 = q2 @ k.transpose(-1, -2)
        
        w1 = weight.mean(dim=1)
        w2 = weight2.mean(dim=1)
        print(w1.shape, w2.shape)
        
        mask = mask.reshape(batch_size, 1, sequence_length, 1).expand_as(weight2)
        # print(mask.shape, weight2.shape)
        mask1 = mask2 = mask
        mask1 = ~mask1
        weight.masked_fill_(mask1, -1000)
        weight2.masked_fill_(mask2, -1000)
        
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight /= math.sqrt(module.d_head)
        weight2 /= math.sqrt(module.d_head)
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = F.softmax(weight, dim=-1)
        weight2 = F.softmax(weight2, dim=-1)
        
        
        
        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        output1 = weight @ v
        output2 = weight2 @ v
        
        output = (output1 + output2)/2
        
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
        output = output.transpose(1, 2)
        
        # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = output.reshape(input_shape)
        
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = module.out_proj(output).contiguous()

        # (Batch_Size, Seq_Len_Q, Dim_Q)
        return output
    
    handles = []
    for name, module in model.named_modules():
        # if 'BGAttention' in name:
        if isinstance(module, BGAttention):
            # print('Hooked',module.named_modules)
            handle = module.register_forward_hook(forward_hook)
            handles.append(handle)
    with torch.no_grad():
        x = model(data_input)
    for handle in handles:
        handle.remove()
        
    return x
    
if __name__ == '__main__':
    model = PathFinder()
    # cal
    def cal_model_size(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)/1024/1024
    print(cal_model_size(model))
    x = torch.randn(3,1,64,64)
    x2 = torch.zeros_like(x)
    x2[:,:,0,6] = 1
    x2[0,:,3,8] = 1
    x2[0,:,10,9] = 1
    
    
    # 
    x = torch.cat([x, x2], dim=1)
    print(x.shape)
    
    
    
    
    
    
    # y = model(x)
    y = wrapper_forward_hook(model, x)
    print(y.shape)