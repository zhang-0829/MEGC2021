
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


    # ori_path: 原始CAS数据集地址
    # tar_path: 目标文件地址
    ori_path = './longVideoFaceCropped'
    tar_path = './longVideoFaceCropped_apex_{0}'
    path = './CAS-data.xlsx'


    sheet_names = list(pd.read_excel(path, sheet_name=None).keys())
    df3 = pd.read_excel(path, sheet_name=sheet_names[-1], header=None)
    df3 = df3.set_index(1)

    df2 = pd.read_excel(path, sheet_name=sheet_names[1], header=None)
    df2 = df2.drop_duplicates()[[1, 2]].set_index(2)

    subject_id = df2.loc[x[0]].values[0]
    path = ori_path+'/'+subject_id+'/'+subject_id[1:]+'_0'+str(df3.loc[x['videoId']][0])+df3.loc[x['videoId']][2]
    start_index = x[2]
    apex_index = x[3]
    end_index = x[4]

    if start_index <= 0 or apex_index <= 0 or end_index <= 0:
        return

    target_path = tar_path.format('micro') if x[7] == 'micro-expression' else tar_path.format('macro')
    target_path = target_path+'/'+subject_id+'/'+x[1]

    move_pics(path, target_path, start_index, apex_index)

def move_pics(path,target_path,start_index,apex_index):
    mkdir(target_path)
    for i in [start_index,apex_index]:
        pic_name = 'img_{0}.jpg'.format(i)
        shutil.copyfile(path+'/'+pic_name,target_path+'/'+pic_name)



if __name__ == '__main__':


    path = './CAS-data.xlsx'

    df = pd.read_excel(path, header=None)
    df['videoId'] = df[1].apply(lambda x: x.split('_')[0])
    print(df)
    df.apply(process_single_exp,axis=1)


    print("---------------OVER------------- !!! ")
    pass
