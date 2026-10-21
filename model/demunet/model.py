import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import relu
from torch.nn import LayerNorm,Linear,Dropout,BatchNorm2d
import torch.nn.functional as F

from torch.nn import TransformerEncoderLayer
from torch.nn import TransformerEncoder
import math
from .mamba import Mamba, MambaConfig
# U 形主干特征宽度；输入波段数和注意力维度在 MyModel 中按参数确定。
oc=256

def position_embeddings(n_pos_vec, dim):
    position_embedding = torch.nn.Embedding(n_pos_vec.numel(), dim)
    torch.nn.init.constant_(position_embedding.weight, 0.)
    return position_embedding

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_convs=nn.Sequential(
            nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.double_convs(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up,self).__init__()

        if bilinear:

            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:

            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

        self.up_match_channels=nn.Conv2d(in_channels,out_channels,kernel_size=1)
 
    def forward(self,x1,x2):
        x1=self.up(x1)
        x1=self.up_match_channels(x1)

        diffY=x2.size()[2]-x1.size()[2]
        diffX= x2.size()[3]-x1.size()[3]

        x1=F.pad(x1,[diffX//2,diffX-diffX//2,diffY//2,diffY-diffY//2])

        x=torch.cat([x2,x1],dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(OutConv,self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=1)
 
    def forward(self,x):
        return self.conv(x)

class MambaLayers(nn.Module):
    def __init__(self, d_model, n_layers):
        super(MambaLayers, self).__init__()
        self.mamba = Mamba(MambaConfig(d_model=d_model, n_layers=n_layers))

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.reshape(bs, c, -1)
        x = x.permute(0, 2, 1)
        x = self.mamba(x)
        x = x.permute(0, 2, 1)
        x = x.reshape(bs, c, h, w)
        return x

class EncoderLayers(nn.Module):
    def __init__(self,encoder_in,num_encoder_layers=3,dim_feedforward=384,nhead=8,dropout=0.1):
        super(EncoderLayers, self).__init__()
      #  encoder_layer = TransformerEncoderLayer(encoder_in, nhead, dim_feedforward, dropout,norm_first=False)
        encoder_layer = TransformerEncoderLayer(
    encoder_in, nhead, dim_feedforward, dropout,
    norm_first=False, batch_first=True
)
        encoder_norm =LayerNorm(encoder_in)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.reshape(bs, c, -1)
        x = x.permute(0, 2, 1)

        x = self.encoder(x)
        
        x = x.permute(0, 2, 1)
        x = x.reshape(bs, c, h, w)
        return x    

class Pooling(nn.Module):
    """
    @article{ref-vit,
    title={An image is worth 16x16 words: Transformers for image recognition at scale},
    author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, 
            Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
    journal={arXiv preprint arXiv:2010.11929},
    year={2020}
    }
    """
    def __init__(self, pool: str = "mean"):
        super().__init__()
        if pool not in ["mean", "cls"]:
            raise ValueError("pool must be one of {mean, cls}")

        self.pool_fn = self.mean_pool if pool == "mean" else self.cls_pool

    def mean_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def cls_pool(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_fn(x)


class Classifier(nn.Module):

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(in_features=dim, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  
    
class MyModel(torch.nn.Module):
    """Demunet patch 分类：ic 为输入波段数，ix 为严格小于 ic 的最大 8 倍数。

    Houston: ic=144, ix=136; PU: ic=103, ix=96.
    batch_size 是位置嵌入可支持的最大批量，验证/测试可使用更小的尾批。
    """

    def __init__(self,batch_size=32,bilinear=True,*,in_channels=144,num_classes=15,patch_size=11):
        super(MyModel, self).__init__()
        if not isinstance(in_channels, int) or in_channels <= 8:
            raise ValueError('Demunet in_channels must be an integer greater than 8')
        if not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError('Demunet num_classes must be a positive integer')
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('Demunet batch_size must be a positive integer')
        if not isinstance(patch_size, int) or patch_size < 5 or patch_size % 2 == 0:
            raise ValueError('Demunet patch_size must be odd and at least 5')
        ic = in_channels
        ix = ((ic - 1) // 8) * 8
        self.ic = ic
        self.ix = ix
        self.patch_size = patch_size
        self.max_batch_size = batch_size
        
        self.inc= DoubleConv(ic,oc)
        self.down1= Down(oc,512)
        self.down2=Down(512,1024)

        self.up2= Up(512,oc,bilinear)
        self.up1= Up(1024,512,bilinear)

        self.dropout=Dropout(0.5)

        self.mamba = MambaLayers(d_model=ic,n_layers=1)
        self.mamba2 = MambaLayers(d_model=ic,n_layers=1)

        self.encoder1 = EncoderLayers(encoder_in=ix,num_encoder_layers=1,dim_feedforward=512,nhead=8)
        
        self.position_embedding = position_embeddings(torch.arange(batch_size*ic*patch_size*patch_size), 1)
        self.BN=nn.BatchNorm2d(ic)
        self.lmd = nn.Parameter(torch.tensor(0.5))

        self.pool = Pooling(pool="mean")
        self.classifier = Classifier(dim=oc, num_classes=num_classes)

        self.bn1 = nn.BatchNorm2d(ic)
        self.conv1 = nn.Conv2d(ic, 64, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, ic, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(ic, ix, kernel_size=1, padding=0)
        self.ver_conv = nn.Conv2d(ix, ic, kernel_size=1, padding=0)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError('Demunet expects [batch, bands, height, width]')
        batch_size, c, h, w= x.shape
        # 从已有层读取通道数，也兼容尚无 self.ic/self.ix 的旧完整模型对象。
        input_channels = self.conv.in_channels
        attention_channels = self.conv.out_channels
        configured_patch_size = getattr(self, 'patch_size', h)
        max_batch_size = getattr(
            self, 'max_batch_size',
            self.position_embedding.num_embeddings // (input_channels * h * w),
        )
        if c != input_channels:
            raise ValueError(f'Demunet expects {input_channels} bands, got {c}')
        if (h, w) != (configured_patch_size, configured_patch_size):
            raise ValueError(f'Demunet expects a {configured_patch_size}x{configured_patch_size} patch')
        if batch_size > max_batch_size:
            raise ValueError(f'Demunet batch size {batch_size} exceeds configured maximum {max_batch_size}')

        x1=x.permute(0, 1, 3, 2)
        x1=self.mamba2(x1) 
        # print(x1.shape)
        x1=x1.permute(0, 1, 3, 2)
        x=self.mamba(x)
        # print(x.shape)
        x=(1-self.lmd)*x+self.lmd*x1
        # print(x.shape)

        x=self.BN(x)
        x1=self.conv1(self.relu(self.bn1(x)))
        x1 = self.conv2(self.relu(self.bn2(x1)))
        x1 = self.conv3(self.relu(self.bn3(x1)))

        x = self.conv(x)
        position_ids = torch.arange(batch_size*attention_channels*h*w, dtype=torch.long, device=x.device)
        position_embeds = self.position_embedding(position_ids).reshape(batch_size, attention_channels,h,w)
        x=x+position_embeds


        x=self.encoder1(x)
        x = self.ver_conv(x)

        x = x + x1

        x1= self.inc(x)
        
        x2= self.down1(x1)

        x3= self.down2(x2)
      
        x= self.up1(x3,x2)

        x=self.up2(x,x1)
        x=self.dropout(x)

        x=x.reshape(batch_size,-1,h*w)

        x=x.permute(0,2,1)

        x=self.pool(x)

        x = self.classifier(x)

        return x
