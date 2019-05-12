# -*- coding: utf-8 -*-
"""
Created on Wed May  8 17:00:38 2019

@author: 86156
"""

import torch
import numpy as np
from model import ImgTxtModel
from dataset import UrbanDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import sys
import os

def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = np.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()
    

def main(test_list_path,input_mean=[0.468, 0.537, 0.621], input_std=[0.173, 0.157, 0.139]):
    model = ImgTxtModel()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image_size = model.input_image_size
    
    check_point = torch.load('snapshot/__model_best.pth.tar')
    model.load_state_dict(check_point['state_dict'])
    print('-'*20)
    print('load check_point __model_best.pth.tar successful !')
    print('-'*20)  
    
    
    test_loader = DataLoader(UrbanDataset(test_list_path,
                                    image_transfrom = transforms.Compose([
#                                            transforms.CenterCrop(image_size),
                                            transforms.Resize(image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(input_mean,input_std)]),
                                        mode='test'),
                            batch_size=16,shuffle=False, num_workers=0)    
    # set to eval, drop out will be disabled
    # remember to set eval , it also change bn
    model.eval()
    torch.set_grad_enabled(False)
    total_step = len(test_loader)
    predicitions = []
    for i, (input, __) in enumerate(test_loader): 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input_var = [x.to(device) for x in input]
        output =  model(input_var)
        _, pred = torch.max(output, 1)
        predicitions.extend(pred)
        view_bar('val:',i,total_step)    
    
    if not os.path.exists("result/"):
        os.mkdir("result/")
        
    with open("result/result.txt", "w+") as f:
        for index, prediction in enumerate(predicitions):
            f.write("%s \t %03d\n"%(str(index).zfill(6), prediction+1))
            
    print("\n--------test success!------")
    
    
if __name__ == '__main__':
    print('hell0, test!')
    test_list_path = 'data_list/test_list.txt'
    main(test_list_path)