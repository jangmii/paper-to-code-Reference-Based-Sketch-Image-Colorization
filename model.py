import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import os
import numpy as np
import torch
import torch.nn as nn
import math

class Generator(nn.Module):
    
    def __init__(self, sketch_channels=1, reference_channels=3, LR_negative_slope=0.2):
        super(Generator, self).__init__()
        
        self.encoder_sketch = Encoder(sketch_channels)
        self.encoder_reference = Encoder(reference_channels)
        self.scft = SCFT(sketch_channels, reference_channels)
        self.resblock = ResBlock(992, 992)
        self.unet_decoder = UNetDecoder()
    
    def forward(self, sketch_img, reference_img):
        
        # encoder 
        Vs, F = self.encoder_sketch(sketch_img)
        Vr, _ = self.encoder_reference(reference_img)
        
        # scft
        c, quary, key, value = self.scft(Vs,Vr)
        
        # resblock
        c_out = self.resblock(c)
        
        # unet decoder
        I_gt = self.unet_decoder(torch.cat((c,c_out),dim=1), F)

        return I_gt, quary, key, value

class Encoder(nn.Module):
    
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        
        def CL2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, LR_negative_slope=0.2):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=bias)]
            layers += [nn.LeakyReLU(LR_negative_slope)]
            cbr = nn.Sequential(*layers)
            return cbr
        
        # conv_layer
        self.conv1 = CL2d(in_channels,16)
        self.conv2 = CL2d(16,16)
        self.conv3 = CL2d(16,32,stride=2)
        self.conv4 = CL2d(32,32)
        self.conv5 = CL2d(32,64,stride=2)
        self.conv6 = CL2d(64,64)
        self.conv7 = CL2d(64,128,stride=2)
        self.conv8 = CL2d(128,128)
        self.conv9 = CL2d(128,256,stride=2)
        self.conv10 = CL2d(256,256)
        
        # downsample_layer
        self.downsample1 = nn.AvgPool2d(kernel_size=16, stride=16)
        self.downsample2 = nn.AvgPool2d(kernel_size=8, stride=8)
        self.downsample3 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.downsample4 = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):

        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        f5 = self.conv5(f4)
        f6 = self.conv6(f5)
        f7 = self.conv7(f6)
        f8 = self.conv8(f7)
        f9 = self.conv9(f8)
        f10 = self.conv10(f9)
        
        F = [f9, f8, f7, f6, f5, f4, f3, f2 ,f1]
        
        v1 = self.downsample1(f1)
        v2 = self.downsample1(f2)
        v3 = self.downsample2(f3)
        v4 = self.downsample2(f4)
        v5 = self.downsample3(f5)
        v6 = self.downsample3(f6)
        v7 = self.downsample4(f7)
        v8 = self.downsample4(f8)

        V = torch.cat((v1,v2,v3,v4,v5,v6,v7,v8,f9,f10), dim=1)
        V = torch.reshape(V,(V.size(0),V.size(1),V.size(2)*V.size(3)))
        V = torch.transpose(V,1,2)
        
        return V,F

class SCFT(nn.Module):
    
    def __init__(self, sketch_channels, reference_channels, dv=992):
        super(SCFT, self).__init__()
        
        self.dv = torch.tensor(dv).float()
        
        self.w_q = nn.Linear(dv,dv)
        self.w_k = nn.Linear(dv,dv)
        self.w_v = nn.Linear(dv,dv)
        
    def forward(self, Vs, Vr):

        quary = self.w_q(Vs)
        key = self.w_k(Vr)
        value = self.w_v(Vr)

        c = torch.add(self.scaled_dot_product(quary,key,value), Vs)
        
        c = torch.transpose(c,1,2)
        c = torch.reshape(c,(c.size(0),c.size(1),16,16))
        
        return c, quary, key, value

    # https://www.quantumdl.com/entry/11%EC%A3%BC%EC%B0%A82-Attention-is-All-You-Need-Transformer
    def scaled_dot_product(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim = -1)

        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value)

class UNetDecoder(nn.Module):
    def __init__(self):
        super(UNetDecoder, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr
        

        self.dec5_1 = CBR2d(in_channels=992+992, out_channels=256)
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=512+128, out_channels=128)
        self.dec4_1 = CBR2d(in_channels=128+128, out_channels=128)
        self.unpool3 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=128+64, out_channels=64)
        self.dec3_1 = CBR2d(in_channels=64+64, out_channels=64)
        self.unpool2 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=64+32, out_channels=32)
        self.dec2_1 = CBR2d(in_channels=32+32, out_channels=32)
        self.unpool1 = nn.ConvTranspose2d(in_channels=32, out_channels=32,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=32+16, out_channels=16)
        self.dec1_1 = CBR2d(in_channels=16+16, out_channels=16)

        self.fc = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, F):
        
        dec5_1 = self.dec5_1(x)
        unpool4 = self.unpool4(torch.cat((dec5_1,F[0]),dim=1))

        dec4_2 = self.dec4_2(torch.cat((unpool4,F[1]),dim=1))
        dec4_1 = self.dec4_1(torch.cat((dec4_2,F[2]),dim=1))
        unpool3 = self.unpool3(dec4_1)

        dec3_2 = self.dec3_2(torch.cat((unpool3,F[3]),dim=1))
        dec3_1 = self.dec3_1(torch.cat((dec3_2,F[4]),dim=1))
        unpool2 = self.unpool2(dec3_1)

        dec2_2 = self.dec2_2(torch.cat((unpool2,F[5]),dim=1))
        dec2_1 = self.dec2_1(torch.cat((dec2_2,F[6]),dim=1))
        unpool1 = self.unpool1(dec2_1)
        
        dec1_2 = self.dec1_2(torch.cat((unpool1,F[7]),dim=1))
        dec1_1 = self.dec1_1(torch.cat((dec1_2, F[8]),dim=1))

        x = self.fc(dec1_1)
        
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        
        def block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]

            cbr = nn.Sequential(*layers)

            return cbr
        
        self.block_1 = block(in_channels,out_channels)
        self.block_2 = block(out_channels,out_channels)
        self.block_3 = block(out_channels,out_channels)
        self.block_4 = block(out_channels,out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        
        # block 1
        residual = x
        out = self.block_1(x)
        out += residual
        out = self.relu(out)
        
        # block 2
        residual = out
        out = self.block_2(x)
        out += residual
        out = self.relu(out)
        
        # block 3
        residual = out
        out = self.block_3(x)
        out += residual
        out = self.relu(out)
        
        # block 4
        residual = out
        out = self.block_4(x)
        out += residual
        out = self.relu(out)
        
        return out

# https://github.com/meliketoy/LSGAN.pytorch/blob/master/networks/Discriminator.py
# LSGAN Discriminator
class Discriminator(nn.Module):
    def __init__(self, ndf, nChannels):
        super(Discriminator, self).__init__()
        # input : (batch * nChannels * image width * image height)
        # Discriminator will be consisted with a series of convolution networks

        self.layer1 = nn.Sequential(
            # Input size : input image with dimension (nChannels)*64*64
            # Output size: output feature vector with (ndf)*32*32
            nn.Conv2d(
                in_channels = nChannels,
                out_channels = ndf,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias = False
            ),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer2 = nn.Sequential(
            # Input size : input feature vector with (ndf)*32*32
            # Output size: output feature vector with (ndf*2)*16*16
            nn.Conv2d(
                in_channels = ndf,
                out_channels = ndf*2,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias = False
            ),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer3 = nn.Sequential(
            # Input size : input feature vector with (ndf*2)*16*16
            # Output size: output feature vector with (ndf*4)*8*8
            nn.Conv2d(
                in_channels = ndf*2,
                out_channels = ndf*4,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias = False
            ),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer4 = nn.Sequential(
            # Input size : input feature vector with (ndf*4)*8*8
            # Output size: output feature vector with (ndf*8)*4*4
            nn.Conv2d(
                in_channels = ndf*4,
                out_channels = ndf*8,
                kernel_size = 4,
                stride = 2,
                padding = 1,
                bias = False
            ),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.layer5 = nn.Sequential(
            # Input size : input feature vector with (ndf*8)*4*4
            # Output size: output probability of fake/real image
            nn.Conv2d(
                in_channels = ndf*8,
                out_channels = 1,
                kernel_size = 4,
                stride = 1,
                padding = 0,
                bias = False
            ),
            # nn.Sigmoid() -- Replaced with Least Square Loss
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out
