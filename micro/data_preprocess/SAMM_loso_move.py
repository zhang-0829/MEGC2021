import numpy as np
import os
import pandas as pd
from os.path import basename
import  shutil
import cv2
import re
import dlib


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  
        os.makedirs(path)  
        print("---  new folder...  ---")
        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")

def optical_flow(img1,img2):
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    final = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return final



# 作用读取video的视频帧
# 输入：frame上级文件夹的地址
# 输出：jpg后缀的Series
def read_videoFrames(path):
    frames = os.listdir(path)
    sorted_frames = sorted(frames, key=lambda x: re.findall("\d+", x.split('_')[-1])[0])
    return sorted_frames


def getData(exp_path):
    compute_optical = []
    leye = []
    mouth = []
    reye = []
    for frame in read_videoFrames(exp_path):
        img_path = os.path.join(exp_path, frame)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        print(img_path)
        compute_optical.append(img)
    print('----------------------------')
    compute_optical = np.array(compute_optical)

    detect = face_detector(compute_optical[0,], 1)
    shape = face_pose_predictor(compute_optical[0,], detect[0])
    # 左眼
    x11 = shape.part(17).x - 12
    x12 = shape.part(21).x + 12
    y11 = shape.part(19).y - 12
    y12 = shape.part(41).y + 12
    # 右眼
    x21 = shape.part(22).x - 12
    x22 = shape.part(26).x + 12
    y21 = shape.part(24).y - 12
    y22 = shape.part(46).y + 12
    # 嘴巴
    x31 = shape.part(48).x - 12
    x32 = shape.part(54).x + 12
    y31 = shape.part(50).y - 12
    y32 = shape.part(58).y + 12

    # 计算两帧之间的光流
    final = optical_flow(compute_optical[0,], compute_optical[1,])
    # 提取roi区域
    leye = final[y11:y12, x11:x12, :]
    reye = final[y21:y22, x21:x22, :]
    mouth = final[y31:y32, x31:x32, :]

    leye = cv2.resize(leye, (70, 50))
    reye = cv2.resize(reye, (70, 50))
    mouth = cv2.resize(mouth, (70, 40))

    return leye , reye , mouth , final


if __name__ == '__main__':


    model = './shape_predictor_68_face_landmarks.dat'

    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(model)

    # ori_path: CAS onset-apex 数据集地址
    # loso_path: 目标文件地址

    ori_path = './{0}'
    loso_path = './0711_loso_data/{0}'
    types = ['micro','natural']
    data_names = ['train_left_eye', 'train_right_eye', 'train_mouth', 'train_face', 'train_label','valid_left_eye', 'valid_right_eye', 'valid_mouth', 'valid_face', 'valid_label']

    xlsx_path = './SAMM-data.xlsx'

    df = pd.read_excel(xlsx_path, skiprows=9)
    df = df[df.columns[:8]]
    print(df)
    subject_ids = df['Subject'].drop_duplicates().values

    mkdir(ori_path.format('loso_data'))

    for subject_id in subject_ids:
        print('--------------------------------')
        print('该轮mask id为：')
        print(subject_id)

        subject_id = str(subject_id).zfill(3)
        mkdir(loso_path.format(subject_id))


        datas = []
        for i in range(10):
            datas.append([])
        for type_id, type in enumerate(types):
            data_paths = os.listdir(ori_path.format(type))

            if type_id == 1:
                char = ' '
            else:
                char = '_'
            for data_path in data_paths:
                l, r, m, f= getData(os.path.join(ori_path.format(type),data_path))
                if data_path.split(char)[0] != subject_id:
                    datas[0].append(l)
                    datas[1].append(r)
                    datas[2].append(m)
                    datas[3].append(f)
                    datas[4].append(type_id)
                else:
                    datas[5].append(l)
                    datas[6].append(r)
                    datas[7].append(m)
                    datas[8].append(f)
                    datas[9].append(type_id)
                    print('not bingo')
                print(data_path)

        for i,x in enumerate(data_names):
            print(len(datas[i]))
            temp_dir = loso_path.format(subject_id)
            np.save(temp_dir+'/{0}_{1}.npy'.format(subject_id,x), datas[i])

        print('单个subject_id结束')
