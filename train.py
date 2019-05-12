# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:09:07 2019

@author: 86156
"""

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import shutil
import numpy as np
import sys
from model import ImgTxtModel
from dataset import UrbanDataset
cuda_device_id = '0'

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        target = torch.zeros_like(input).scatter_(1, target.view(-1,1), 1)
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()
        
def train(dataloader, model, criterion, optimazer, epoch, grad_cilp=True):
    losses = AverageMeter()
    accuracy = AverageMeter()
    
    model.train()
    torch.set_grad_enabled(True)    

    i_step = 50
    total_step = len(dataloader)
    
    for i, (input, target) in enumerate(dataloader):
        if i == len(dataloader)-1:
            break
        
        device = torch.device("cuda:"+cuda_device_id if torch.cuda.is_available() else "cpu")
        input_var = [x.to(device) for x in input]
        batch_size = input_var[0].size(0)
        target = target.to(device)
        
        model.txt_model.hidden_state = [x.to(device) for x in model.txt_model.init_hidden()]
        
        output =  model(input_var)
        #penalization AAT - I
#        attT = att.transpose(1,2)
#        identity = torch.eye(att.size(1)).to(device)
#        identity = identity.unsqueeze(0).expand(batch_size,att.size(1),att.size(1))
#        penal = model.l2_matrix_norm(att@attT - identity)
#        loss = criterion(output, target) + (1.0 * penal/batch_size)
        
        loss = criterion(output, target)
        loss.backward()
        if grad_cilp:
            torch.nn.utils.clip_grad_norm_(model.parameters(),0.5)
        optimazer.step()
        optimazer.zero_grad()
        losses.update(loss.item(),batch_size)
        
        _, pred = torch.max(output, 1)
        accuracy.update(torch.sum(pred == target.data).item()/batch_size, batch_size)
    
        if (i+1) % i_step == 0:
            print('Train --- epoch: {:02d}, step: [{:04d}/{:04d}], loss: {:.4f},\
                  acc: {:.2f}'.format(epoch, i, total_step, losses.avg, accuracy.avg))
            
def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = np.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()
   
def validation(dataloader, model, criterion, epoch):
    model.eval()  
    torch.set_grad_enabled(False)
    
    loss_sum,acc_sum = 0,0
    total_step = len(dataloader)

    for i, (input, target) in enumerate(dataloader):
        if i == len(dataloader)-1:
            break       
        
        device = torch.device("cuda:"+cuda_device_id if torch.cuda.is_available() else "cpu")
        input_var = [x.to(device) for x in input]
        target = target.to(device)
        batch_size = input_var[0].size(0)
        
        output =  model(input_var)
        loss = criterion(output, target)
        loss_sum += loss.item() * batch_size
        
        _, pred = torch.max(output, 1)
        acc_sum += torch.sum(pred == target.data)
        
        view_bar('val:',i,total_step)
        
    loss_avg = loss_sum / (total_step * batch_size)
    acc_avg = acc_sum.double() / (total_step * batch_size)
    print('\n')
    print('Val --- epoch: loss: {:.4f}, acc: {:.2f}'.format(loss_avg, acc_avg))  
    return acc_avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    filename = 'snapshot/__' + str(state['epoch']) + '__' + filename
    torch.save(state, filename)
    if is_best:
        best_name = 'snapshot/__' + 'model_best.pth.tar'
        shutil.copyfile(filename, best_name)
    
        
    
def main(run_epoch, train_list_file, val_list_file, val_frequency, resume, batch_size,\
         input_mean=[0.468, 0.537, 0.621], input_std=[0.173, 0.157, 0.139], mode='train'):

    model = ImgTxtModel()
    device = torch.device("cuda:"+cuda_device_id if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    image_size = model.input_image_size
    # setup dataloads
    
    train_loader = DataLoader(UrbanDataset(train_list_file,
                                    image_transfrom = transforms.Compose([
                                            transforms.RandomResizedCrop(image_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(input_mean,input_std)])),
                            batch_size=batch_size,shuffle=True, num_workers=0)
    
    val_loader = DataLoader(UrbanDataset(val_list_file,
                                    image_transfrom = transforms.Compose([
                                            transforms.CenterCrop(image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize(input_mean,input_std)])),
                            batch_size=batch_size,shuffle=True, num_workers=0)    
    
    if resume:
        check_point = torch.load('snapshot/__model_best.pth.tar')
        best_acc = check_point['best_acc']
        start_epoch = check_point['epoch']
        model.load_state_dict(check_point['state_dict'])
        print('-'*20)
        print('load check_point __model_best.pth.tar successful !')
        print('-'*20)
    else:
        best_acc = 0
        start_epoch = 0
        
    # judge validation or train
    if mode == 'train':
        # build loss function and optimazer
#        criterion = torch.nn.CrossEntropyLoss().to(device)
        criterion = FocalLoss().to(device)
        optimazer = torch.optim.SGD(model.parameters(), lr=0.008, momentum=0.9)
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimazer, step_size=3, gamma=0.9)
        print('----- start training -----')
        for epoch in range(start_epoch, start_epoch+run_epoch):
            train(train_loader, model, criterion, optimazer, epoch)
            exp_lr_scheduler.step()
#            print('current lr is : {:4f}'.format(optimazer.param_groups[0]['lr']))

            if (epoch+1) % val_frequency == 0:
                model.eval()
                acc = validation(val_loader, model, criterion, epoch)
                
                is_best = acc > best_acc
                # save model
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'lr': optimazer.param_groups[-1]['lr'],
                }, is_best)                
                
    else:
        model.eval()
        validation(val_loader, model, criterion, 0)


if __name__ == '__main__':

    train_root_path = 'data_list/train_list.txt'
    val_root_path = 'data_list/val_list.txt'
    start_epoch = 0
    run_epoch = 30
    val_frequency = 1
    batch_size = 16
    main(run_epoch, train_root_path, val_root_path, val_frequency, resume=False, batch_size=batch_size)
    