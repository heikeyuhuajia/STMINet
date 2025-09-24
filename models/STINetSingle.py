import torch
import torch.nn as nn
from .resnet import resnet18, resnet50
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
from timm.models.layers import trunc_normal_
import cv2

# AANet
from models.utilsFromAANet import *

# uper
# 24.3.31
from mmcv.cnn import ConvModule
from mmseg.registry import MODELS
from mmseg.models.utils import resize
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.decode_heads.psp_head import PPM

# Swin
# 24.4.13
from mmseg.models.backbones import SwinTransformer

# ln
from mmseg.models.utils import LayerNorm2d
from mmcv.cnn import (ConvModule,  Conv2d, build_norm_layer,
                      ConvModule, build_activation_layer)
from mmengine.model import BaseModule, Sequential
from mmcv.cnn.bricks.drop import build_dropout
 
from einops import rearrange, repeat
from models.ema import EMA

def init_weights(m):
    """
    Initialize weights of layers using Kaiming Normal (He et al.) as argument of "Apply" function of
    "nn.Module"
    :param m: Layer to initialize
    :return: None
    """
    if isinstance(m, nn.Conv2d):
        '''
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        trunc_normal_(m.weight, std=math.sqrt(1.0/fan_in)/.87962566103423978)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        '''
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)
        
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

        
class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        # print("++",x.size()[1],self.inp_dim,x.size()[1],self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class decode(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, out_channel,norm_layer=nn.BatchNorm2d):
        super(decode, self).__init__()
        self.conv_d1 = nn.Conv2d(in_channel_down, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(in_channel_left, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channel*2, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn3 = norm_layer(out_channel)

    def forward(self, left, down):
        down_mask = self.conv_d1(down)
        left_mask = self.conv_l(left)
        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear')
            z1 = F.relu(left_mask * down_, inplace=True)
        else:
            z1 = F.relu(left_mask * down, inplace=True)

        if down_mask.size()[2:] != left.size()[2:]:
            down_mask = F.interpolate(down_mask, size=left.size()[2:], mode='bilinear')

        z2 = F.relu(down_mask * left, inplace=True)

        out = torch.cat((z1, z2), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
class SpacetimeAttSingle_v25(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., space_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5   # ？？

        self.qd = nn.Linear(dim, dim//2, bias=qkv_bias)
        self.qu = nn.Linear(dim, dim//2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.space_ratio = space_ratio    # sr_v1=8, 4, 4/2 ,1;  sr_v2=8, 4, 4/2 ,1; 
        if space_ratio > 1:
            self.act = nn.GELU()
            if space_ratio==16:
                self.s_d1 = nn.Conv2d(64, 64, kernel_size=8, stride=8)
                self.norm_d1 = nn.LayerNorm(64)
                self.s_d2 = nn.Conv2d(64, 64, kernel_size=4, stride=4)
                self.norm_d2 = nn.LayerNorm(64)
                self.s_u1 = nn.Conv2d(64, 64, kernel_size=8, stride=8)
                self.norm_u1 = nn.LayerNorm(64)
                self.s_u2 = nn.Conv2d(64, 64, kernel_size=4, stride=4)
                self.norm_u2 = nn.LayerNorm(64)

            if space_ratio==8:
                self.s_d1 = nn.Conv2d(128, 128, kernel_size=4, stride=4)
                self.norm_d1 = nn.LayerNorm(128)
                self.s_d2 = nn.Conv2d(128, 128, kernel_size=2, stride=2)
                self.norm_d2 = nn.LayerNorm(128)
                self.s_u1 = nn.Conv2d(128, 128, kernel_size=4, stride=4)
                self.norm_u1 = nn.LayerNorm(128)
                self.s_u2 = nn.Conv2d(128, 128, kernel_size=2, stride=2)
                self.norm_u2 = nn.LayerNorm(128)

            if space_ratio==4:

                self.s_d1 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
                self.norm_d1 = nn.LayerNorm(256)
                self.s_d2 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
                self.norm_d2 = nn.LayerNorm(256)
                self.s_u1 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
                self.norm_u1 = nn.LayerNorm(256)
                self.s_u2 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
                self.norm_u2 = nn.LayerNorm(256)

            if space_ratio==2:
                self.s_d1 = nn.Conv2d(512, 512, kernel_size=1, stride=1)
                self.norm_d1 = nn.LayerNorm(512)
                self.s_d2 = nn.Conv2d(512, 512, kernel_size=1, stride=1)
                self.norm_d2 = nn.LayerNorm(512)
                self.s_u1 = nn.Conv2d(512, 512, kernel_size=1, stride=1)
                self.norm_u1 = nn.LayerNorm(512)
                self.s_u2 = nn.Conv2d(512, 512, kernel_size=1, stride=1)
                self.norm_u2 = nn.LayerNorm(512)

            self.kv_d1 = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv_d2 = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv_u1 = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv_u2 = nn.Linear(dim, dim, bias=qkv_bias)
            
        else:
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x_d, x_u):
        B1, C1, H1, W1 = x_d.shape
        B2, C2, H2, W2 = x_u.shape

        x_d_ = x_d.permute(0, 2, 3, 1).reshape(B1, H1*W1, -1)
        x_u_ = x_u.permute(0, 2, 3, 1).reshape(B2, H2*W2, -1)

        if self.space_ratio == 16:
            self.num_heads = 2
            q_d = self.qd(x_d_).reshape(B1, H1*W1, self.num_heads, 32 // self.num_heads).permute(0, 2, 1, 3)   
            q_u = self.qu(x_u_).reshape(B2, H2*W2, self.num_heads, 32 // self.num_heads).permute(0, 2, 1, 3)
            q = torch.cat([q_d,q_u],3)

            x_d_1 = self.act(self.norm_d1(self.s_d1(x_d).reshape(B2, 64, -1).permute(0, 2, 1)))
            x_d_2 = self.act(self.norm_d2(self.s_d2(x_d).reshape(B2, 64, -1).permute(0, 2, 1)))
            x_u_1 = self.act(self.norm_u1(self.s_u1(x_u).reshape(B2, 64, -1).permute(0, 2, 1)))
            x_u_2 = self.act(self.norm_u2(self.s_u2(x_u).reshape(B2, 64, -1).permute(0, 2, 1)))
            
            kv_d_1 = self.kv_d1(x_d_1).reshape(B2, -1, 2, self.num_heads//2, 64 // self.num_heads).permute(2, 0, 3, 1, 4)
            kv_d_2 = self.kv_d2(x_d_2).reshape(B2, -1, 2, self.num_heads//2, 64 // self.num_heads).permute(2, 0, 3, 1, 4)
            kv_u_1 = self.kv_u1(x_u_1).reshape(B2, -1, 2, self.num_heads//2, 64 // self.num_heads).permute(2, 0, 3, 1, 4)
            kv_u_2 = self.kv_u2(x_u_2).reshape(B2, -1, 2, self.num_heads//2, 64 // self.num_heads).permute(2, 0, 3, 1, 4)

            k_d_1, v_d_1 = kv_d_1[0], kv_d_1[1]
            k_d_2, v_d_2 = kv_d_2[0], kv_d_2[1]
            k_u_1, v_u_1 = kv_u_1[0], kv_u_1[1]
            k_u_2, v_u_2 = kv_u_2[0], kv_u_2[1]

            # --------------------------------------
            attn1 = (q[:, :self.num_heads//2] @ k_u_1.transpose(-2, -1)) * self.scale
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)
            x1 = (attn1 @ v_u_1).transpose(1, 2).reshape(B1, H1*W1, 64//2)  

            attn2 = (q[:, self.num_heads//2:] @ k_u_2.transpose(-2, -1)) * self.scale
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)
            x2 = (attn2 @ v_u_2).transpose(1, 2).reshape(B1, H1*W1, 64//2)
            
            attn3 = (q[:, :self.num_heads//2] @ k_d_1.transpose(-2, -1)) * self.scale
            attn3 = attn3.softmax(dim=-1)
            attn3 = self.attn_drop(attn3)

            x3 = (attn3 @ v_d_1).transpose(1, 2).reshape(B2, H2*W2, 64//2)  

            attn4 = (q[:, self.num_heads//2:] @ k_d_2.transpose(-2, -1)) * self.scale
            attn4 = attn4.softmax(dim=-1)
            attn4 = self.attn_drop(attn4)
           
            x4 = (attn4 @ v_d_2).transpose(1, 2).reshape(B2, H2*W2, 64//2)
            # -------------------------------------
            out_d = torch.cat([x1,x2], dim=-1)
            out_u = torch.cat([x3,x4], dim=-1)

        elif self.space_ratio == 8:
            self.num_heads = 4 

            q_d = self.qd(x_d_).reshape(B1, H1*W1, self.num_heads, 64 // self.num_heads).permute(0, 2, 1, 3)    # 未来改进1 生成Q通过linear
            q_u = self.qu(x_u_).reshape(B2, H2*W2, self.num_heads, 64 // self.num_heads).permute(0, 2, 1, 3)
            q = torch.cat([q_d,q_u],3)

            x_d_1 = self.act(self.norm_d1(self.s_d1(x_d).reshape(B2, 128, -1).permute(0, 2, 1)))
            x_d_2 = self.act(self.norm_d2(self.s_d2(x_d).reshape(B2, 128, -1).permute(0, 2, 1)))
            x_u_1 = self.act(self.norm_u1(self.s_u1(x_u).reshape(B2, 128, -1).permute(0, 2, 1)))
            x_u_2 = self.act(self.norm_u2(self.s_u2(x_u).reshape(B2, 128, -1).permute(0, 2, 1)))
            
            kv_d_1 = self.kv_d1(x_d_1).reshape(B2, -1, 2, self.num_heads//2, 128 // self.num_heads).permute(2, 0, 3, 1, 4)
            kv_d_2 = self.kv_d2(x_d_2).reshape(B2, -1, 2, self.num_heads//2, 128 // self.num_heads).permute(2, 0, 3, 1, 4)
            kv_u_1 = self.kv_u1(x_u_1).reshape(B2, -1, 2, self.num_heads//2, 128 // self.num_heads).permute(2, 0, 3, 1, 4)
            kv_u_2 = self.kv_u2(x_u_2).reshape(B2, -1, 2, self.num_heads//2, 128 // self.num_heads).permute(2, 0, 3, 1, 4)

            k_d_1, v_d_1 = kv_d_1[0], kv_d_1[1]
            k_d_2, v_d_2 = kv_d_2[0], kv_d_2[1]
            k_u_1, v_u_1 = kv_u_1[0], kv_u_1[1]
            k_u_2, v_u_2 = kv_u_2[0], kv_u_2[1]

            # --------------------------------------
            attn1 = (q[:, :self.num_heads//2] @ k_u_1.transpose(-2, -1)) * self.scale
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)
        
            x1 = (attn1 @ v_u_1).transpose(1, 2).reshape(B1, H1*W1, 128//2)  

            attn2 = (q[:, self.num_heads//2:] @ k_u_2.transpose(-2, -1)) * self.scale
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)
            
            x2 = (attn2 @ v_u_2).transpose(1, 2).reshape(B1, H1*W1, 128//2)
            
            attn3 = (q[:, :self.num_heads//2] @ k_d_1.transpose(-2, -1)) * self.scale
            attn3 = attn3.softmax(dim=-1)
            attn3 = self.attn_drop(attn3)
            
            x3 = (attn3 @ v_d_1).transpose(1, 2).reshape(B2, H2*W2, 128//2)  

            attn4 = (q[:, self.num_heads//2:] @ k_d_2.transpose(-2, -1)) * self.scale
            attn4 = attn4.softmax(dim=-1)
            attn4 = self.attn_drop(attn4)
            
            x4 = (attn4 @ v_d_2).transpose(1, 2).reshape(B2, H2*W2, 128//2)
            # --------------------------------------
            out_d = torch.cat([x1,x2], dim=-1)
            out_u = torch.cat([x3,x4], dim=-1)

        elif self.space_ratio == 4:
            self.num_heads = 8 
           
            q_d = self.qd(x_d_).reshape(B1, H1*W1, self.num_heads, 128 // self.num_heads).permute(0, 2, 1, 3)    # 未来改进1 生成Q通过linear
            q_u = self.qu(x_u_).reshape(B2, H2*W2, self.num_heads, 128 // self.num_heads).permute(0, 2, 1, 3)
            q = torch.cat([q_d,q_u],3)

            x_d_1 = self.act(self.norm_d1(self.s_d1(x_d).reshape(B2, 256, -1).permute(0, 2, 1)))
            x_d_2 = self.act(self.norm_d2(self.s_d2(x_d).reshape(B2, 256, -1).permute(0, 2, 1)))
            x_u_1 = self.act(self.norm_u1(self.s_u1(x_u).reshape(B2, 256, -1).permute(0, 2, 1)))
            x_u_2 = self.act(self.norm_u2(self.s_u2(x_u).reshape(B2, 256, -1).permute(0, 2, 1)))
            
            kv_d_1 = self.kv_d1(x_d_1).reshape(B2, -1, 2, self.num_heads//2, 256 // self.num_heads).permute(2, 0, 3, 1, 4)
            kv_d_2 = self.kv_d2(x_d_2).reshape(B2, -1, 2, self.num_heads//2, 256 // self.num_heads).permute(2, 0, 3, 1, 4)
            kv_u_1 = self.kv_u1(x_u_1).reshape(B2, -1, 2, self.num_heads//2, 256 // self.num_heads).permute(2, 0, 3, 1, 4)
            kv_u_2 = self.kv_u2(x_u_2).reshape(B2, -1, 2, self.num_heads//2, 256 // self.num_heads).permute(2, 0, 3, 1, 4)

            k_d_1, v_d_1 = kv_d_1[0], kv_d_1[1]
            k_d_2, v_d_2 = kv_d_2[0], kv_d_2[1]
            k_u_1, v_u_1 = kv_u_1[0], kv_u_1[1]
            k_u_2, v_u_2 = kv_u_2[0], kv_u_2[1]

            # --------------------------------------
            attn1 = (q[:, :self.num_heads//2] @ k_u_1.transpose(-2, -1)) * self.scale
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)
            
            x1 = (attn1 @ v_u_1).transpose(1, 2).reshape(B1, H1*W1, 256//2)  

            attn2 = (q[:, self.num_heads//2:] @ k_u_2.transpose(-2, -1)) * self.scale
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)
            
            x2 = (attn2 @ v_u_2).transpose(1, 2).reshape(B1, H1*W1, 256//2)
            
            attn3 = (q[:, :self.num_heads//2] @ k_d_1.transpose(-2, -1)) * self.scale
            attn3 = attn3.softmax(dim=-1)
            attn3 = self.attn_drop(attn3)
            
            x3 = (attn3 @ v_d_1).transpose(1, 2).reshape(B2, H2*W2, 256//2)  

            attn4 = (q[:, self.num_heads//2:] @ k_d_2.transpose(-2, -1)) * self.scale
            attn4 = attn4.softmax(dim=-1)
            attn4 = self.attn_drop(attn4)
            
            x4 = (attn4 @ v_d_2).transpose(1, 2).reshape(B2, H2*W2, 256//2)
            # --------------------------------------
            out_d = torch.cat([x1,x2], dim=-1)
            out_u = torch.cat([x3,x4], dim=-1)

        elif self.space_ratio == 2:
            self.num_heads = 16 

            q_d = self.qd(x_d_).reshape(B1, H1*W1, self.num_heads, 256 // self.num_heads).permute(0, 2, 1, 3)   
            q_u = self.qu(x_u_).reshape(B2, H2*W2, self.num_heads, 256 // self.num_heads).permute(0, 2, 1, 3)
            q = torch.cat([q_d,q_u],3)

            x_d_1 = self.act(self.norm_d1(self.s_d1(x_d).reshape(B2, 512, -1).permute(0, 2, 1)))
            x_d_2 = self.act(self.norm_d2(self.s_d2(x_d).reshape(B2, 512, -1).permute(0, 2, 1)))
            x_u_1 = self.act(self.norm_u1(self.s_u1(x_u).reshape(B2, 512, -1).permute(0, 2, 1)))
            x_u_2 = self.act(self.norm_u2(self.s_u2(x_u).reshape(B2, 512, -1).permute(0, 2, 1)))
            
            kv_d_1 = self.kv_d1(x_d_1).reshape(B2, -1, 2, self.num_heads//2, 512 // self.num_heads).permute(2, 0, 3, 1, 4)
            kv_d_2 = self.kv_d2(x_d_2).reshape(B2, -1, 2, self.num_heads//2, 512 // self.num_heads).permute(2, 0, 3, 1, 4)
            kv_u_1 = self.kv_u1(x_u_1).reshape(B2, -1, 2, self.num_heads//2, 512 // self.num_heads).permute(2, 0, 3, 1, 4)
            kv_u_2 = self.kv_u2(x_u_2).reshape(B2, -1, 2, self.num_heads//2, 512 // self.num_heads).permute(2, 0, 3, 1, 4)

            k_d_1, v_d_1 = kv_d_1[0], kv_d_1[1]
            k_d_2, v_d_2 = kv_d_2[0], kv_d_2[1]
            k_u_1, v_u_1 = kv_u_1[0], kv_u_1[1]
            k_u_2, v_u_2 = kv_u_2[0], kv_u_2[1]

            # --------------------------------------
            attn1 = (q[:, :self.num_heads//2] @ k_u_1.transpose(-2, -1)) * self.scale
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)
            
            x1 = (attn1 @ v_u_1).transpose(1, 2).reshape(B1, H1*W1, 512//2)  

            attn2 = (q[:, self.num_heads//2:] @ k_u_2.transpose(-2, -1)) * self.scale
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)
           
            x2 = (attn2 @ v_u_2).transpose(1, 2).reshape(B1, H1*W1, 512//2)
            
            attn3 = (q[:, :self.num_heads//2] @ k_d_1.transpose(-2, -1)) * self.scale
            attn3 = attn3.softmax(dim=-1)
            attn3 = self.attn_drop(attn3)
            
            x3 = (attn3 @ v_d_1).transpose(1, 2).reshape(B2, H2*W2, 512//2)  

            attn4 = (q[:, self.num_heads//2:] @ k_d_2.transpose(-2, -1)) * self.scale
            attn4 = attn4.softmax(dim=-1)
            attn4 = self.attn_drop(attn4)
           
            x4 = (attn4 @ v_d_2).transpose(1, 2).reshape(B2, H2*W2, 512//2)
            # --------------------------------------
            out_d = torch.cat([x1,x2], dim=-1)
            out_u = torch.cat([x3,x4], dim=-1)

        out_d = out_d.view(B1, C1, H1, W1)
        out_u = out_u.view(B2, C2, H2, W2)
        return x_d + self.gamma1 * out_d, x_u + self.gamma2 * out_u
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
class CoordAtt_STMI(nn.Module):
    # from Lighter
    #
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt_STMI, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        a_h = a_h.expand(-1,-1,h,w)
        a_w = a_w.expand(-1, -1, h, w)

        out = identity * a_w * a_h
        return out

###############   STMINet-up   ###############
class STINetSingle_v25_1(nn.Module):


    def __init__(self, num_classes=2, drop_rate=0.2, normal_init=True, pretrained=False, show_Feature_Maps=False):
        super(STINetSingle_v25_1, self).__init__()

        self.show_Feature_Maps = True
        self.bid = 120

        self.resnet = resnet18()
        self.resnet.load_state_dict(torch.load('./pretrained/resnet18-5c106cde.pth'))

        self.sta1 = SpacetimeAttSingle_v25(64, attn_drop=0., space_ratio=16)
        self.sta2 = SpacetimeAttSingle_v25(128, attn_drop=0., space_ratio=8)
        self.sta3 = SpacetimeAttSingle_v25(256, attn_drop=0., space_ratio=4)
        self.sta4 = SpacetimeAttSingle_v25(512, attn_drop=0., space_ratio=2)

        self.Ambiguity01 = AmbiguityRefinementModule(512, 512, upsize=1)
        self.Ambiguity02 = AmbiguityRefinementModule(256, 256, upsize=1)
        self.Ambiguity03 = AmbiguityRefinementModule(128, 128, upsize=1)
        self.Ambiguity04 = AmbiguityRefinementModule(64, 64, upsize=1)

        self.conv_cat0 = nn.Sequential(nn.Conv2d(1024, 512, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU(),
                                       CoordAtt_STMI(512, 512))
        self.conv_cat1 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       CoordAtt_STMI(256, 256))
        self.conv_cat2 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(128),
                                       nn.ReLU(),
                                       CoordAtt_STMI(128, 128))
        self.conv_cat3 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1, bias=False),
                                       nn.BatchNorm2d(64),
                                       nn.ReLU(),
                                       CoordAtt_STMI(64, 64))

        self.Translayer1_am = BasicConv2d(512, 256, 1)
        self.fam22_am = decode(256, 256, 256)

        self.Translayer2_am = BasicConv2d(256, 128, 1)
        self.fam32_am = decode(128, 128, 128)

        self.Translayer3_am = BasicConv2d(128, 64, 1)
        self.fam43_am = decode(64, 64, 64)
       
        self.Translayer1_1 = BasicConv2d(512, 256, 1)
        self.fam22_1 = decode(256, 256, 256)
        self.Translayer2_1 = BasicConv2d(256, 128, 1)
        self.fam32_1 = decode(128, 128, 128)  
        self.Translayer3_1 = BasicConv2d(128, 64, 1)
        self.fam43_1 = decode(64, 64, 64)  

        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsamplex8 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsamplex16 = nn.Upsample(scale_factor=16, mode='bilinear')

        self.final = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
        )
        self.finalam = nn.Sequential(
            Conv(64, 32, 3, bn=True, relu=True),
            Conv(32, num_classes, 3, bn=False, relu=False)
        )
        self.final_2 = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )
        self.final2_am = nn.Sequential(
            Conv(128, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )
        self.final_3 = nn.Sequential(
            Conv(256, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )
        self.final3_am = nn.Sequential(
            Conv(256, 64, 3, bn=True, relu=True),
            Conv(64, num_classes, 3, bn=False, relu=False)
        )
        if normal_init:
            self.init_weights()

    def forward(self, imgs1, imgs2, bid=None, labels=None):  # hot
        c0 = self.resnet.conv1(imgs1)  # 1/2
        c0 = self.resnet.bn1(c0)
        c0 = self.resnet.relu(c0)
        c1 = self.resnet.maxpool(c0)  # 1/4 64
        c1 = self.resnet.layer1(c1)  # 1/4 64
        c2 = self.resnet.layer2(c1)  # 1/8 32
        c3 = self.resnet.layer3(c2)  # 1/16 16
        c4 = self.resnet.layer4(c3)  # 1/32 8

        c0_img2 = self.resnet.conv1(imgs2)
        c0_img2 = self.resnet.bn1(c0_img2)
        c0_img2 = self.resnet.relu(c0_img2)
        c1_img2 = self.resnet.maxpool(c0_img2)
        c1_img2 = self.resnet.layer1(c1_img2)
        c2_img2 = self.resnet.layer2(c1_img2)
        c3_img2 = self.resnet.layer3(c2_img2)
        c4_img2 = self.resnet.layer4(c3_img2)

        # inter
        st_result0, st_result_img0 = self.sta4(c4, c4_img2)
        st_result1, st_result_img1 = self.sta3(c3, c3_img2)
        st_result2, st_result_img2 = self.sta2(c2, c2_img2) 
        st_result3, st_result_img3 = self.sta1(c1, c1_img2)

        # am
        st_am0 = self.Ambiguity01(st_result0, st_result_img0)
        st_am1 = self.Ambiguity02(st_result1, st_result_img1)
        st_am2 = self.Ambiguity03(st_result2, st_result_img2)
        st_am3 = self.Ambiguity04(st_result3, st_result_img3)

        # fuse
        out2_am = self.fam22_am(st_am1, self.Translayer1_am(st_am0))  
        out3_am = self.fam32_am(st_am2, self.Translayer2_am(out2_am))
        out4_am = self.fam43_am(st_am3, self.Translayer3_am(out3_am))

        st_ca0 = self.conv_cat0(torch.cat([st_result0, st_result_img0], 1))
        st_ca1 = self.conv_cat1(torch.cat([st_result1, st_result_img1], 1))
        st_ca2 = self.conv_cat2(torch.cat([st_result2, st_result_img2], 1))
        st_ca3 = self.conv_cat3(torch.cat([st_result3, st_result_img3], 1))

        out2 = self.fam22_1(st_ca1, self.Translayer1_1(st_ca0))  
        out3 = self.fam32_1(st_ca2, self.Translayer2_1(out2))
        out4 = self.fam43_1(st_ca3, self.Translayer3_1(out3))

        out_1 = self.final(self.upsamplex4(out4))  # 1/4 
        out_am = self.finalam(self.upsamplex4(out4_am))  

        out_1_2 = self.final_2(self.upsamplex8(out3))  # 1/8
        out_2_am = self.final2_am(self.upsamplex8(out3_am))

        out_3 = self.final_3(self.upsamplex16(out2))  # 1/16
        out_3_am = self.final3_am(self.upsamplex16(out2_am))
        
        return out_1, out_am, out_1_2, out_2_am, out_3, out_3_am  

    def init_weights(self):
        self.sta1.apply(init_weights)
        self.sta2.apply(init_weights)
        self.sta3.apply(init_weights)
        self.sta4.apply(init_weights)

        self.Ambiguity01.apply(init_weights)
        self.Ambiguity02.apply(init_weights)
        self.Ambiguity03.apply(init_weights)
        self.Ambiguity04.apply(init_weights)

        self.fam22_am.apply(init_weights)
        self.Translayer1_am.apply(init_weights)
        self.fam32_am.apply(init_weights)
        self.Translayer2_am.apply(init_weights)
        self.fam43_am.apply(init_weights)
        self.Translayer3_am.apply(init_weights)

        self.conv_cat0.apply(init_weights)
        self.conv_cat1.apply(init_weights)
        self.conv_cat2.apply(init_weights)
        self.conv_cat3.apply(init_weights)

        self.fam22_1.apply(init_weights)
        self.Translayer1_1.apply(init_weights)
        self.fam32_1.apply(init_weights)
        self.Translayer2_1.apply(init_weights)
        self.fam43_1.apply(init_weights)
        self.Translayer3_1.apply(init_weights)

        self.final.apply(init_weights)
        self.final_2.apply(init_weights)
        self.final_3.apply(init_weights)

        self.finalam.apply(init_weights)
        self.final2_am.apply(init_weights)
        self.final3_am.apply(init_weights)
