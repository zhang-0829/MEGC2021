from keras.models import load_model
from tensorflow.keras.models import load_model
import os
import cv2
import numpy as np
import numpy
import pandas as pd
import dlib


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def single_video_detected(path):
    micro_frame_length = 110
    micro_step = 110
    micro_data = pred_single_video(path, micro_frame_length, micro_step, 0)
    return [[micro_data[i],0] for i in range(3)]


# 利用模型进行预测
# 返回预测种类
def pred_single_frames(paths,type):
     data = getData(paths)
     predict = model.predict(data)
     print(predict)
     p = 0.9
     pred_type = np.argmax(predict)

     if pred_type == 0:
         max_p = np.max(predict)
         if max_p >= p:
             return pred_type
         else:
             return 2
     else:
         return 2

def optical_flow(img1,img2):
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY) 
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    final = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return final

# 作用：数据预处理，处理单张图片以便预测
# 输入：1）视频帧路径List；2）当前数据帧状态
def getData(exp_path):

    compute_optical = []
    for img_path in exp_path:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        print(img_path)
        compute_optical.append(img)
    print('----------------------------')
    compute_optical = np.array(compute_optical)

    detect = face_detector(compute_optical[0,], 1)
    shape = face_pose_predictor(compute_optical[0,], detect[0])
    # #左眼
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

    ##计算两帧之间的光流
    final = optical_flow(compute_optical[0,], compute_optical[1,])
    # 提取roi区域
    leye = final[y11:y12, x11:x12, :]
    reye = final[y21:y22, x21:x22, :]
    mouth = final[y31:y32, x31:x32, :]

    leye = cv2.resize(leye, (70, 50))
    reye = cv2.resize(reye, (70, 50))
    mouth = cv2.resize(mouth, (70, 40))

    leye = np.array([leye])
    reye = np.array([reye])
    mouth = np.array([mouth])
    face = np.array([final])

    return [leye,reye,mouth,face]


# 判断该帧是否为TP
# 返回值：True or Fasle
def judge_TP(nowIndexStart, nowIndexEnd, now_exp_startIndex, now_exp_endIndex, thresh):

    length_1 = nowIndexEnd - nowIndexStart + 1
    length_2 = now_exp_endIndex - now_exp_startIndex + 1

    frame_list = [nowIndexStart, nowIndexEnd, now_exp_startIndex, now_exp_endIndex]

    denominator = max(frame_list) - min(frame_list)

    numerator = length_1 + length_2 - denominator

    print('左起始：' + str(now_exp_startIndex) + ' 当前帧起始：' + str(nowIndexStart))
    print('分子：' + str(numerator) + ' 分母：' + str(denominator))
    if numerator / denominator >= thresh:
        return True
    else:
        return False


# 作用：分析单个视频，此时要分别分析宏表情和微表情的相关数据
# 输入：1)单个视频的地址 2）单次使用的frame的长度 3）视频帧步长step 4）当前判断的表情类型
# 输出：TP、FP、TN、Precise、Recall、F1-Score
def pred_single_video(loc, frame_length, step , type):

    expression_type = ['i','a']


    M_singleVideo = 0
    N_singleVideo = 0
    A_singleVideo = 0

    info = loc.split('/')[-1].split('_')
    print(info)


    df_exp_datas = df[(df['Subject'] == int(info[0])) & (df['Inducement Code']== int(info[1])) & (df['Type'].apply(lambda x : x[1] == expression_type[type]))].iloc[:,[3,5,7]]

    print('信息：')
    print(df_exp_datas)
    if df_exp_datas.shape[0] == 0:
        return M_singleVideo, N_singleVideo, A_singleVideo

    now_exp_index = 0
    exp_length = df_exp_datas.shape[0]
    M_singleVideo = exp_length
    now_exp_startIndex = int(df_exp_datas.iloc[now_exp_index][0])
    now_exp_endIndex = int(df_exp_datas.iloc[now_exp_index][1])

    print(now_exp_startIndex, now_exp_endIndex, type)

    s_video_frames = read_videoFrames(loc)
    length = len(s_video_frames)
    start_index = 0

    global df_log


    last_p_start = -1
    last_p_end = -1

    while start_index <= length - frame_length:

        paths = [loc + '/' + x for x in [s_video_frames[start_index], s_video_frames[start_index + frame_length - 1]]]
        pred_type = pred_single_frames(paths, type)

        if pred_type == type:
            if last_p_start == -1:
                last_p_start = start_index
                last_p_end = start_index + frame_length - 1
            elif start_index <= last_p_end:
                last_p_end = start_index + frame_length - 1
            else:
                res = judge_TP(last_p_start, last_p_end, now_exp_startIndex, now_exp_endIndex, 0.5)
                # TP
                if res:
                    print(start_index)
                    A_singleVideo += 1

                    temp_row = {'Video_ID': video_id, 'GT_onset': now_exp_startIndex, 'GT_offset': now_exp_endIndex,
                                'Predicted_onset': last_p_start, 'Predicted_offset': last_p_end,
                                'Type': 'TP'}

                    df_log = df_log.append(temp_row, ignore_index=True)

                    if now_exp_index < exp_length - 1:
                        now_exp_index += 1
                        now_exp_startIndex = int(df_exp_datas.iloc[now_exp_index][0])
                        now_exp_endIndex = int(df_exp_datas.iloc[now_exp_index][1])
                    else:
                        pass
                # FP
                else:
                    temp_row = {'Video_ID': video_id, 'GT_onset': '-', 'GT_offset': '-',
                                'Predicted_onset': last_p_start, 'Predicted_offset': last_p_end,
                                'Type': 'FP'}

                    df_log = df_log.append(temp_row, ignore_index=True)
                N_singleVideo += 1
                last_p_start = start_index
                last_p_end = start_index + frame_length - 1

        else:
            if last_p_start > now_exp_endIndex:
                if now_exp_index < exp_length - 1:

                    temp_row = {'Video_ID': video_id, 'GT_onset': now_exp_startIndex, 'GT_offset': now_exp_endIndex,
                                'Predicted_onset': '-', 'Predicted_offset': '-',
                                'Type': 'FN'}

                    df_log = df_log.append(temp_row, ignore_index=True)

                    now_exp_index += 1
                    now_exp_startIndex = int(df_exp_datas.iloc[now_exp_index][0])
                    now_exp_endIndex = int(df_exp_datas.iloc[now_exp_index][1])
                else:
                    pass

        start_index += step

    if last_p_start != -1:

        res = judge_TP(last_p_start, last_p_end, now_exp_startIndex, now_exp_endIndex, 0.5)
        # TP
        if res:
            print(start_index)
            A_singleVideo += 1

            temp_row = {'Video_ID': video_id, 'GT_onset': now_exp_startIndex, 'GT_offset': now_exp_endIndex,
                        'Predicted_onset': last_p_start, 'Predicted_offset': last_p_end,
                        'Type': 'TP'}

            df_log = df_log.append(temp_row, ignore_index=True)

        # FP
        else:
            temp_row = {'Video_ID': video_id, 'GT_onset': '-', 'GT_offset': '-',
                        'Predicted_onset': last_p_start, 'Predicted_offset': last_p_end,
                        'Type': 'FP'}

            df_log = df_log.append(temp_row, ignore_index=True)
        N_singleVideo += 1
    else:
        temp_row = {'Video_ID': video_id, 'GT_onset': now_exp_startIndex, 'GT_offset': now_exp_endIndex,
                    'Predicted_onset': '-', 'Predicted_offset': '-',
                    'Type': 'FN'}

        df_log = df_log.append(temp_row, ignore_index=True)

    return M_singleVideo, N_singleVideo, A_singleVideo


# 作用读取video的视频帧
# 输入：frame上级文件夹的地址
# 输出：jpg后缀的Series
def read_videoFrames(path):
    frames = os.listdir(path)
    sorted_frames = sorted(frames, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return sorted_frames



# 功能：计算指标
# 输入：A,M,N
# 输出：recall, precision, F1_score
def recall_precision_f1(A_clips, M_clips, N_clips):
    if M_clips != 0:
        recall = float(A_clips) / float(M_clips)  # A / M
    else:
        recall = 0

    if N_clips != 0:
        precision = float(A_clips) / float(N_clips)  # A / N
    else:
        precision = 0

    if (recall + precision) != 0:
        F1_score = 2 * recall * precision / (recall + precision)
    else:
        F1_score = 0

    return recall, precision, F1_score


# 功能：遍历检测所有受试者的视频
def all_video_detected(path):
    video_num = 0
    M = [0, 0]
    N = [0, 0]
    A = [0, 0]

    paths = os.listdir(path)
    paths = sorted(paths, key=lambda x: (int(x.split('_')[0]),int(x.split('_')[1])))


    for video in paths:
        video_path = os.path.join(path, video)
        print('----遍历每个video---')

        global video_id
        video_id = video

        global subject_id

        temp_subject_id = video.split('_')[0]

        if subject_id != temp_subject_id:
            global model
            model = load_model(model_path.format(temp_subject_id, temp_subject_id))
            subject_id = temp_subject_id

        video_num += 1
        print(video_path)
        M_temp , N_temp , A_temp = single_video_detected(video_path)
        for i in range(2):
            M[i] += M_temp[i]
            N[i] += N_temp[i]
            A[i] += A_temp[i]
        print(M_temp)
        print(N_temp)
        print(A_temp)
        # break

    print('总视频数量：')
    print(video_num)

    # 生成log文件
    global df_log
    df_log = df_log.set_index('Video_ID')
    file_name = './samm_log.csv'
    df_log.to_csv(file_name, sep='\t')

    return M,N,A



# 功能：打印指标信息
# 输入：M,N,A
# 输出：无
def print_info(M,N,A):
    recall,precision,F1_score = recall_precision_f1(A,M,N)
    print('M:' + str(M))
    print('N:' + str(N))
    print('A:' + str(A))
    print('recall:' + str(recall))
    print('precision:' + str(precision))
    print('F1_score:' + str(F1_score))

if __name__ == '__main__':


    #path代表预处理后的SAMM数据库数据地址
    path = './SAMM_longvideos'

    data_path = './SAMM-data.xlsx'

    df = pd.read_excel(data_path, skiprows = 9)
    df = df[df.columns[:8]]

    global video_id
    video_id = 0

    subject_id = '000'

    model_path = './0711_loso_data/{1}.h5'


    global df_log
    df_log = pd.DataFrame(columns=('Video_ID', 'GT_onset', 'GT_offset', 'Predicted_onset', 'Predicted_offset', 'Type'))


    landmarks_path = './shape_predictor_68_face_landmarks.dat'

    face_detector = dlib.get_frontal_face_detector()
    face_pose_predictor = dlib.shape_predictor(landmarks_path)

    M, N, A = all_video_detected(path)
    recall = [0,0]
    precision = [0, 0]
    F1_score = [0, 0]

    infos = ['Micro指标:','Macro指标:']

    for i in range(2):
        print(infos[i])
        print_info(M[i],N[i],A[i])

    print('全部数据汇总:')
    M_total = sum(M)
    N_total = sum(N)
    A_total = sum(A)
    print_info(M_total, N_total, A_total)


