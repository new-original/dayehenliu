# -*- coding: utf-8 -*-
"""
Created on Tue May  7 08:35:39 2019

@author: 86156
"""

from torch import nn
import torch
from models import DenseNet
from models.SelfAttention import StructuredSelfAttention

class ImgTxtModel(nn.Module):
    def __init__(self,img_growth_rate=12, img_block_config=[16,16,16], \
                 txt_growth_rate=12, txt_block_config=[16,16], efficient=True, \
                 num_classes=9):
        super(ImgTxtModel, self).__init__()
        
        self.input_image_size = 96
        self.img_model = DenseNet(
            growth_rate=img_growth_rate,
            block_config=img_block_config,
            num_classes=384,
            efficient=efficient)
        
        self.txt_model = StructuredSelfAttention(use_clssify=False)
        self.fc = nn.Linear(384+128, num_classes)
        

    def forward(self, input):
        img_feature = self.img_model(input[0])
        txt_feature,attention = self.txt_model(input[1])
        max_feature = torch.cat((img_feature, txt_feature), 1)
        out = self.fc(max_feature)
        return out


if __name__ == '__main__':
    print('hello')
    basemodel = ImgTxtModel()
    print(basemodel)