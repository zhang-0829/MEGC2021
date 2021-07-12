
import numpy as np
import os
import pandas as pd
from os.path import basename

import  shutil


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  
        os.makedirs(path)  
        print("---  new folder...  ---")
        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")


def process_single_exp(x):


    # ori_path: 原始SAMM数据集地址
    # tar_path: 目标文件地址
    ori_path = './SAMM_longvideos'
    tar_path = './SAMM_longvideos_apex_{0}'


    path = ori_path+'/{0}_{1}'.format(str(x[0]).zfill(3),x[2])
    type = (x[-1])
    start_index = int(x[3])
    apex_index = int(x[4])
    end_index = int(x[5])

    if start_index <= 0 or apex_index <= 0 or end_index <= 0:
        return

    target_path = tar_path.format('micro') if type[1] == 'i' else tar_path.format('macro')
    target_path = target_path+'/'+x[1]

    move_pics(path, target_path, start_index, apex_index)

def move_pics(path,target_path,start_index,apex_index):
    mkdir(target_path)

    pics = os.listdir(path)

    pics.sort(key = lambda x:int(x.split('_')[-1].split('.')[0]))


    pic_length = len(str(len(pics)))


    max_pic_id = int(pics[-1].split('_')[-1].split('.')[0])


    pic_length = 4 if pic_length < 5 else pic_length

    prefix_path = path.split('/')[-1]

    if start_index < max_pic_id:
        pic_name = '{0}_{1}.jpg'.format(prefix_path, str(start_index).zfill(pic_length))
        shutil.copyfile(path + '/' + pic_name, target_path + '/' + pic_name)
    if apex_index < max_pic_id:
        pic_name = '{0}_{1}.jpg'.format(prefix_path, str(apex_index).zfill(pic_length))
        shutil.copyfile(path + '/' + pic_name, target_path + '/' + pic_name)

    print('单个表情完成move！')



if __name__ == '__main__':


    path = './SAMM-data.xlsx'

    df = pd.read_excel(path, skiprows = 9)
    df = df[df.columns[:8]]
    print(df)
    df.apply(process_single_exp,axis=1)

    print("---------------OVER------------- !!! ")
    pass