import tensorflow.keras.layers as tfkl
import tensorflow.keras as tfk
import os
import random
import keras
import pickle 
import math
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler,normalize,MinMaxScaler
from keras.preprocessing import sequence
from keras.models import Sequential
import tensorflow as tf

scaler=MinMaxScaler()
BATCH_SIZE=128
PATH="/home/sounak/Desktop/BTP/Dataset/BIT_Database_New/english_modified/Train/"
force_window_size=11
window_size=30
stride=15

file_list=[]
true_value=[]
max_len=0
for i in range(134):
    paths=PATH+str(i+1)+'/'
    csv_files=[f for f in os.listdir(paths) if (f.endswith('.csv'))]
    for j in range(len(csv_files)):
        csv_files[j]=paths+csv_files[j]
    for files in csv_files:
        folder_name=files.split('.')[0]+'/'
        new_folder_name=folder_name+'PLGF_'+str(force_window_size)+'_window_size_per_point/PLGF_Descriptor_Lines_'+str(window_size)+'_window_size_'+str(stride)+'_stride/'
        file_names=[f for f in os.listdir(new_folder_name) if (f.endswith('.csv'))]
        for j in range(len(file_names)):
            file_names[j]=new_folder_name+file_names[j]
        for file_name in file_names:
            file_list.append(file_name)
            true_value.append(i)
print(len(file_list))
c=list(zip(file_list, true_value))
random.shuffle(c)
final_Paths, y_train = zip(*c)
for file_path in final_Paths:
    data=pd.read_csv(file_path,header=None)
    data=np.asarray(data)
    max_len=max(max_len, data.shape[0])
print(max_len)

#y_train=keras.utils.to_categorical(y_train, num_classes=134)

x_train=[]

for i, file_path in enumerate(final_Paths):
    data=pd.read_csv(file_path,header=None)
    data=np.asarray(data)
    data=scaler.fit_transform(data)
    zero=np.zeros((1, 6))
    while(data.shape[0]<max_len):
        data=np.insert(data, 0, zero, axis=0)
    x_train.append(data)
       


x_train=np.asarray(x_train)
print(x_train.shape)
print(np.amax(x_train))
print(np.amin(x_train))

class lstm_bottleneck(tf.keras.layers.Layer):
    def __init__(self, lstm_units, time_steps, **kwargs):
        self.lstm_units = lstm_units
        self.time_steps = time_steps
        self.lstm_layer = tfkl.LSTM(lstm_units, return_sequences=False)
        self.repeat_layer = tfkl.RepeatVector(time_steps)
        super(lstm_bottleneck, self).__init__(**kwargs)
    def call(self, inputs):
        return self.repeat_layer(self.lstm_layer(inputs))
    def compute_mask(self, inputs, mask=None):
        return mask

time_steps = max_len
n_features = 6
input_layer = tfkl.Input(shape=(time_steps, n_features))
x = tfk.layers.Masking(mask_value=0)(input_layer)
x = tfkl.LSTM(25, return_sequences=True)(x)
x = tfkl.LSTM(50, return_sequences=True)(x)
x = lstm_bottleneck(lstm_units=100, time_steps=time_steps)(x)
x = tfkl.LSTM(50, return_sequences=True)(x)
x = tfkl.LSTM(25, return_sequences=True)(x)
x = tfkl.LSTM(12, return_sequences=True)(x)
x = tfk.layers.Dense(n_features)(x)
lstm_ae = tfk.models.Model(inputs=input_layer, outputs=x)
lstm_ae.compile(optimizer='adam', loss='mse')
print(lstm_ae.summary())

chk = ModelCheckpoint('/home/sounak/Desktop/BTP/CodeFiles/BIT_Database_New/autoencoder_best_seq_len_bs_20_with_masking_weights_tf2.h5', verbose=1, monitor='val_loss', save_best_only=True, mode="min", save_weights_only=True)

lstm_ae.fit(x_train, x_train, batch_size=20, callbacks=[chk], epochs=1000, validation_split=0.1, shuffle=True)
