import numpy as np
import os
import  numpy as np
from  sklearn.model_selection  import  LeaveOneOut
import cv2
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D,Concatenate, MaxPooling2D, Dense,Reshape,LSTM, Flatten, Dropout, ZeroPadding3D ,Convolution2D,Activation,BatchNormalization
from keras.callbacks import ReduceLROnPlateau
import np
import numpy
import cv2
import os
import tensorflow as tf
from keras.utils import np_utils
import os
import cv2
import dlib
import pandas as pd
import numpy
from keras.utils import np_utils


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def getModel():
    input1 = Input(shape = (50,70,2))
    x = Convolution2D(8, (5, 5))(input1)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Activation('relu')(x)
    x = Convolution2D(16, (3, 3))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(128)(x)
    x = Dropout(0.5)(x)
    x = Dense(64)(x)


    input2 = Input(shape = (50,70,2))
    y = Convolution2D(8, (5, 5))(input2)
    y = Activation('relu')(y)
    y = MaxPooling2D(pool_size=(3, 3))(y)
    y = Activation('relu')(y)
    y = Convolution2D(16, (3, 3))(y)
    y = Activation('relu')(y)
    y = MaxPooling2D(pool_size=(3, 3))(y)
    y = Activation('relu')(y)
    y = Dropout(0.5)(y)
    y = Flatten()(y)
    y = Dense(128)(y)
    y = Dropout(0.5)(y)
    y = Dense(64)(y)


    input3 = Input(shape = (40,70,2))
    z = Convolution2D(8, (5, 5))(input3)
    z = Activation('relu')(z)
    z = MaxPooling2D(pool_size=(3, 3))(z)
    z = Activation('relu')(z)
    z = Convolution2D(16, (3, 3))(z)
    z = Activation('relu')(z)
    z = MaxPooling2D(pool_size=(3, 3))(z)
    z = Activation('relu')(z)
    z = Dropout(0.5)(z)
    z = Flatten()(z)
    z = Dense(128)(z)
    z = Dropout(0.5)(z)
    z = Dense(100)(z)

    input4 = Input(shape = (224,224,2))
    m = Convolution2D(8, (5, 5))(input4)
    m = Activation('relu')(m)
    m = MaxPooling2D(pool_size=(3, 3))(m)
    m = Activation('relu')(m)
    m = Convolution2D(16, (3, 3))(m)
    m = Activation('relu')(m)
    m = MaxPooling2D(pool_size=(3, 3))(m)
    m = Activation('relu')(m)
    m = Convolution2D(16, (3, 3))(m)
    m = Activation('relu')(m)
    m = MaxPooling2D(pool_size=(3, 3))(m)

    m = Flatten()(m)

    m = Dense(128)(m)
    m = Dropout(0.5)(m)
    m = Dense(64)(m)


    concat = Concatenate(axis = 1)([x,y,z,m])

    dense_5 = Dense(2, )(concat)
    activation = Activation('softmax')(dense_5)
    model = Model(inputs = [input1,input2,input3,input4], outputs = activation)

    return model

if __name__ == '__main__':

    xlsx_path = './CAS-data.xlsx'

    df_sheetnames = pd.read_excel(xlsx_path, sheet_name=None)
    sheet2_name = list(df_sheetnames.keys())[1]
    sheet3_name = list(df_sheetnames.keys())[-1]
    df2 = pd.read_excel(xlsx_path, sheet_name=sheet2_name, header=None)

    subject_ids = df2[0].drop_duplicates().values

    for subject_id in subject_ids:
        print('--------------------------------')
        print('该轮mask id为：')
        print(subject_id)

        subject_id = 's{0}'.format(subject_id)


        Left_eye = numpy.load('./casme_loso/0711_loso_data/{0}/{1}_train_left_eye.npy'.format(subject_id,subject_id))
        Right_eye = numpy.load('./casme_loso/0711_loso_data/{0}/{1}_train_right_eye.npy'.format(subject_id,subject_id))
        Mouth = numpy.load('./casme_loso/0711_loso_data/{0}/{1}_train_mouth.npy'.format(subject_id,subject_id))
        Face = numpy.load('./casme_loso/0711_loso_data/{0}/{1}_train_face.npy'.format(subject_id,subject_id))
        Label = numpy.load('./casme_loso/0711_loso_data/{0}/{1}_train_label.npy'.format(subject_id,subject_id))

        Left_eye_test = numpy.load('./casme_loso/0711_loso_data/{0}/{1}_valid_left_eye.npy'.format(subject_id,subject_id))
        Right_eye_test = numpy.load('./casme_loso/0711_loso_data/{0}/{1}_valid_right_eye.npy'.format(subject_id,subject_id))
        Mouth_test = numpy.load('./casme_loso/0711_loso_data/{0}/{1}_valid_mouth.npy'.format(subject_id,subject_id))
        Face_test = numpy.load('./casme_loso/0711_loso_data/{0}/{1}_valid_face.npy'.format(subject_id,subject_id))
        Label_test = numpy.load('./casme_loso/0711_loso_data/{0}/{1}_valid_label.npy'.format(subject_id,subject_id))

        model = getModel()

        model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

        filepath="./casme_loso/0711_loso_data/{0}/{1}.h5".format(subject_id,subject_id)

        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        callbacks_list = [checkpoint]
        model.summary()

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=20, mode='auto')
        history = model.fit([Left_eye,Right_eye, Mouth,Face], Label, validation_data = ([Left_eye_test,Right_eye_test, Mouth_test,Face_test],Label_test),
                            callbacks=callbacks_list, batch_size =64, nb_epoch = 300, shuffle=True)

        print('--------------------------------')
        print('该轮结束！')