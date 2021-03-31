#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 17:39:17 2021

@author: shayarib
"""


from base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from base import BaseModel
from utils.helpers import initialize_weights, set_trainable
from itertools import chain
from models import resnet


def x2conv(in_channels, out_channels, inner_channels=None):
    ''' This definition performs convolution twice followed by batch normalisation 
    and feeding the output to ReLu activation function '''
    inner_channels = out_channels // 2 if inner_channels is None else inner_channels
    down_conv = nn.Sequential(
        nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(inner_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(inner_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))
    return down_conv


class decoder(nn.Module):
    ''' This class belongs to the decoding segment of the R2Unet architecture
    which performs the upsampling of the input images'''
    def __init__(self, in_channels, out_channels):
        super(decoder, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    ''' In this block, a recurrent network is created which considers the 
    present input and the past input for the computation of output'''
    def __init__(self,out_channels,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.out_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            #combining two inputs
            x1 = self.conv(x+x1) 
        return x1
        
class RRCNN_block(nn.Module):
    '''This class receives input from the recurrent block and further processes 
    it'''
    def __init__(self,in_channels,out_channels,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(out_channels,t=t),
            Recurrent_block(out_channels,t=t)
        )
        
    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class R2UNet(BaseModel):
    
'''This class defines the structure for R2Unet'''
    def __init__(self, num_classes, in_channels=3, out_channels=1, t=2, freeze_bn=False, **_):
        super(R2UNet, self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)
        
        #encoding structure
        self.start_conv = RRCNN_block(in_channels, 64,t)
        self.down1 = RRCNN_block(64, 128,t)
        self.down2 = RRCNN_block(128, 256,t)
        self.down3 = RRCNN_block(256, 512,t)
        self.down4 = RRCNN_block(512, 1024,t)
 
       # In this layer, the size of the image remains constant 
        self.middle_conv = x2conv(1024, 1024)

        #decoding structure
        self.up1 = decoder(1024, 512)
        self.up_RRCNN5= RRCNN_block(1024, 512,t)
        self.up2 = decoder(512, 256)
        self.up_RRCNN4= RRCNN_block(512,256,t)
        self.up3 = decoder(256, 128)
        self.up_RRCNN3= RRCNN_block(256,128,t)
        self.up4 = decoder(128, 64)
        self.up_RRCNN2= RRCNN_block(128, 64,t)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1,stride=1, padding=0)
        self._initialize_weights()

        if freeze_bn:
            self.freeze_bn()

    def _initialize_weights(self):
        ''' Initialising weights to the inputs'''
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x):
        
        #encoding
        e1 = self.start_conv(x)
        
        e2= self.Maxpool(e1)
        e2 = self.down1(e2)
        
        e3= self.Maxpool(e2)
        e3 = self.down2(e3)
        
        e4= self.Maxpool(e3)
        e4 = self.down3(e4)
        
        e5= self.Maxpool(e4)
        e5= self.middle_conv(self.down4(e5))
        
        #decoding 
        d5 = self.Up5(x)
        d5 = torch.cat((e4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((e3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        x = self.final_conv(d2)
        return x

    def get_backbone_params(self):
        # There is no backbone for r2unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()



