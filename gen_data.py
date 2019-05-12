# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:16:50 2019

@author: 86156
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import datetime


def visit2array(table):
    # 用字典查询代替类型转换，可以减少一部分计算时间
    date2position = {}
    datestr2dateint = {}
    str2int = {}
    for i in range(24):
        str2int[str(i).zfill(2)] = i
    
    # 访问记录内的时间从2018年10月1日起，共182天
    # 将日期按日历排列
    for i in range(182):
        date = datetime.date(day=1, month=10, year=2018)+datetime.timedelta(days=i)
        date_int = int(date.__str__().replace("-", ""))
        date2position[date_int] = [i%7, i//7]
        datestr2dateint[str(date_int)] = date_int
        
    strings = table[1]
    init = np.zeros((7, 24, 26))
    for string in strings:
        temp = []
        for item in string.split(','):
            temp.append([item[0:8], item[9:].split("|")])
        for date, visit_lst in temp:
            # dim0: 星期几   dim1: 时间  dim2：第几周  value: 这个时间点的人数  
            x, y = date2position[datestr2dateint[date]]
            for visit in visit_lst: # 统计到访的总人数
                init[x][str2int[visit]][y] += 1
    return init

def text_save(filename, data):#filename为写入CSV文件的路径，data为要写入数据列表.
    with open(filename,'a') as file:
        for i in range(len(data)):
            s = str(data[i]).replace('[','').replace(']','')#去除[],这两行按数据不同，可以选择
            s = s.replace("'",'').replace(',','') +'\n'   #去除单引号，逗号，每行末尾追加换行符
            file.write(s)
    print("{} save successful !".format(filename)) 
    
def text_to_npy(txt_root_path,npy_root_path):
    txt_dir_list = os.listdir(txt_root_path)
    
    if not os.path.exists(npy_root_path):
        os.makedirs(npy_root_path)
    for idx in tqdm(range(len(txt_dir_list))):
        table = pd.read_table(txt_root_path+txt_dir_list[idx], header=None) 
        array = visit2array(table)
        np.save(os.path.join(npy_root_path,txt_dir_list[idx].split('.')[0]+'.npy'), array)


def gen_list(mode='train'):
    if mode=='train':
        img_root_path = 'dataset/train/image/'
        txt_root_path = 'dataset/train/text/'
        npy_root_path = 'dataset/train/text_to_npy/'
        txt_dir_list = os.listdir(txt_root_path)

        img_path_list = [img_root_path + file.split('.')[0].split('_')[1] + '/'+ \
                         file.split('.')[0] + '.jpg '+ file.split('.')[0].split('_')[1] \
                         for file in txt_dir_list]
        
        txt_img_list = [npy_root_path+txt_dir_list[idx].split('.')[0]+'.npy '+img_path_list[idx] \
                        for idx in range(len(txt_dir_list))]
        
        train_list, val_list = train_test_split(txt_img_list, test_size=0.2,random_state=33)
        text_save('data_list/train_list.txt', train_list)
        text_save('data_list/val_list.txt', val_list)
    else:
        img_root_path = 'dataset/test/image/'
        npy_root_path = 'dataset/test/text_to_npy/'        
        txt_dir_list = os.listdir(npy_root_path)
        img_dir_list = os.listdir(img_root_path)
        
        txt_img_list = [npy_root_path+txt_dir_list[ind]+' '+ img_root_path+img_dir_list[ind] for ind in range(len(txt_dir_list))]
        text_save('data_list/test_list.txt', txt_img_list)


def gen_balance_list(mode='train'):
    if mode=='train':
        img_root_path = 'dataset/train/image/'
        txt_root_path = 'dataset/train/text/'
        npy_root_path = 'dataset/train/text_to_npy/'
        txt_dir_list = os.listdir(txt_root_path)

        img_path_list = [img_root_path + file.split('.')[0].split('_')[1] + '/'+ \
                         file.split('.')[0] + '.jpg '+ file.split('.')[0].split('_')[1] \
                         for file in txt_dir_list]
        
        txt_img_list = [npy_root_path+txt_dir_list[idx].split('.')[0]+'.npy '+img_path_list[idx] \
                        for idx in range(len(txt_dir_list))]
        
        train_list, val_list = train_test_split(txt_img_list, test_size=0.2,random_state=33)
        text_save('data_list/train_list.txt', train_list)
        text_save('data_list/val_list.txt', val_list)
        
if __name__ == '__main__':
    print('hello')
    
    img_root_path = '../dataset/train/image/'
    train_list, val_list = [], []
    num_class = 9
    max_length = 3816
    index_change = []
    curr_index = 0    
    
    
#    txt_root_path = 'dataset/train/text/'
#    npy_root_path = 'dataset/train/text_to_npy/'
#    text_to_npy(txt_root_path, npy_root_path)
#    txt_root_path = 'dataset/test/text/'
#    npy_root_path = 'dataset/test/text_to_npy/'
#    text_to_npy(txt_root_path, npy_root_path)
#    gen_list(mode='test')