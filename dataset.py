# -*- coding: utf-8 -*-
"""
Created on Mon May  6 18:10:13 2019

@author: 86156
"""
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import torch

class DataRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def txt_path(self):
        return self._data[0]

    @property
    def img_path(self):
        return self._data[1]

    @property
    def label(self):
        return int(self._data[2])-1
    
    
class UrbanDataset(Dataset):
    def __init__(self, list_file, image_transfrom=None, txt_transform=None, mode='train'):
        self.list_file = list_file
        self.image_transfrom = image_transfrom     
        self.txt_transform = txt_transform
        self.mode = mode
        self._parse_list()
    
    def _parse_list(self):
        self.data_list = [DataRecord(x.strip().split(' ')) for x in open(self.list_file)]    
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        record = self.data_list[index]
        img = Image.open(record.img_path).convert('RGB')
        txt = torch.from_numpy(np.load(record.txt_path).transpose(2, 0, 1)).float()
#        txt = np.random.randn(5,5,3)
        if self.image_transfrom:
            img = self.image_transfrom(img)
        if self.txt_transform:
            txt = self.txt_transform(txt)
        if self.mode == 'train':
            return [img,txt], record.label
        else:
            return [img,txt], ''

def dataset_show():
    list_file = 'data_list/train_list.txt'
    dataset = UrbanDataset(list_file)    
    for i in range(len(dataset)):
        sample = dataset[i]
        print('image size  :', np.array(sample[0][0]).shape)
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        name = 'ant' if sample[1]==0 else 'bee'
        ax.set_title('it is a  #{}'.format(name))
        ax.axis('off')
        plt.imshow(np.array(sample[0][0]))        
        plt.pause(0.001)
        
        if i == 3:
            plt.show()
            break    
        
        
def dataload_show():
    list_file = 'data_list/train_list.txt'
    
    dataset = UrbanDataset(list_file, 
                           image_transfrom = transforms.Compose([
                                   transforms.RandomResizedCrop(224),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor()]))
    
    dataloader = DataLoader(dataset, batch_size=4,
                        shuffle=True, num_workers=0)
    
    for i, (input, target) in enumerate(dataloader):
        if i == 1:
            for i_batch in range(4):
                ax = plt.subplot(1, 4, i + 1)
                plt.tight_layout()
                name = 'ant' if target[i_batch]==0 else 'bee'
                ax.set_title('it is a  #{}'.format(name))
                ax.axis('off')
                plt.imshow(input[0][i_batch].numpy().transpose(1, 2, 0))        
                plt.pause(0.001)
            plt.show()
            break  
        
        
def view_bar(message, num, total):
    import sys
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = np.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()
    
    
def cal_mean_std():
    pop_mean = []
    pop_std0 = []
    list_file = 'data_list/test_list.txt'
    dataset = UrbanDataset(list_file, 
                           image_transfrom = transforms.Compose([
                                   transforms.RandomResizedCrop(224),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor()]), mode='test')
    
    dataloader = DataLoader(dataset, batch_size=16,
                        shuffle=True, num_workers=0)    
    
    for i, (img, label) in enumerate(dataloader):
        # shape (batch_size, 3, height, width)
        numpy_image = img[0].numpy()
    
        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0, 2, 3))
        batch_std0 = np.std(numpy_image, axis=(0, 2, 3))
    
        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)
        view_bar('mean_std----', i, len(dataloader))
    
    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    pop_mean = np.array(pop_mean).mean(axis=0)
    pop_std0 = np.array(pop_std0).mean(axis=0)
    print('\n')
    print(pop_mean, pop_std0)
    
    
    
if __name__ == '__main__':
#    dataload_show()
    cal_mean_std()