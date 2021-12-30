import os
import random
import keras
import pickle 
import math
import pandas as pd
import numpy as np
import tensorflow as tf
from batch_generators import DataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler,normalize
from keras.layers import LSTM, Bidirectional, Dropout, Input, Masking
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

BATCH_SIZE=128
PATH="/home/sounak/Desktop/BTP/Dataset/BIT_Database_New/english_modified/Train/"
window_size=11
substroke_size=15

def mean_square_loss_ignoring_zeros(y_true, y_pred):
    intermediate_tensor = tf.reduce_sum(tf.abs(y_true), 2)
    zero_vector = tf.zeros(shape=(1,1), dtype=tf.float32)
    bool_mask = tf.squeeze(tf.not_equal(intermediate_tensor, zero_vector))
    y_true.set_shape([None, None, None])
    y_pred.set_shape([None, None, None])
    bool_mask.set_shape([None, None, None])
    y_true_masked = tf.boolean_mask(y_true, bool_mask)
    y_pred_masked = tf.boolean_mask(y_pred, bool_mask)
    return K.mean(K.square(y_pred_masked - y_true_masked))
    

file_list=[]
max_len=0
for i in range(134):
    paths=PATH+str(i+1)+'/'
    csv_files=[f for f in os.listdir(paths) if (f.endswith('.csv'))]
    for j in range(len(csv_files)):
        csv_files[j]=paths+csv_files[j]
    for files in csv_files:
        folder_name=files.split('.')[0]+'/'
        new_folder_name=folder_name+'PLGF_'+str(window_size)+'_window_size_per_point/PLGF_Descriptor_Lines_'+str(window_size)+'_per_point/'
        file_names=[f for f in os.listdir(new_folder_name) if (f.endswith('.csv'))]
        for j in range(len(file_names)):
            file_names[j]=new_folder_name+file_names[j]
        for file_name in file_names:
            file_list.append(file_name)

for file_path in file_list:
    data=pd.read_csv(file_path, header=None)
    data=np.asarray(data)
    max_len=max(max_len, data.shape[0])

x_train=[]
for i, file_path in enumerate(file_list):
    data=pd.read_csv(file_path, header=None)
    data=np.asarray(data)
    data=normalize(data, axis=0)
    zero=np.zeros((1, 6))
    zero=np.float32(zero)
    while(data.shape[0]<max_len):
       data=np.insert(data, 0, zero, axis=0)
    x_train.append(data) 

x_train=np.asarray(x_train)
print(x_train.shape)

model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(None, 6)))
model.add(LSTM(100, activation='tanh'))
model.add(RepeatVector(max_len))
model.add(LSTM(6, activation='tanh', return_sequences=True))
model.compile(optimizer='adam', loss=mean_square_loss_ignoring_zeros)

chk = ModelCheckpoint('autoencoder_best_seq_len_max_len.h5', verbose=1, monitor='val_loss', save_best_only=True, mode="min")
model.summary()

model.fit(x_train, x_train, batch_size=32, callbacks=[chk], epochs=1000, shuffle=True)
