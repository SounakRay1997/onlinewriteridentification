import os
import random
import keras
import pickle 
import math
import pandas as pd
import numpy as np
from batch_generators import DataGenerator
from keras.optimizers import SGD
from keras.optimizers import RMSProp
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler,normalize
from keras.layers import LSTM, Bidirectional, Dropout, Input
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense

BATCH_SIZE=128
PATH="/home/sounak/Desktop/BTP/Dataset/BIT_Database_New/english_modified/Train/"
window_size=9
substroke_size=15

file_list=[]
true_value=[]
for i in range(134):
    paths=PATH+str(i+1)+'/'
    csv_files=[f for f in os.listdir(paths) if (f.endswith('.csv'))]
    for j in range(len(csv_files)):
        csv_files[j]=paths+csv_files[j]
    for files in csv_files:
        folder_name=files.split('.')[0]+'/'
        new_folder_name=folder_name+'PLGF_'+str(window_size)+'_window_size/PLGF_Descriptor_Lines_'+str(substroke_size)+'_substrokes/'
        file_names=[f for f in os.listdir(new_folder_name) if (f.endswith('.csv'))]
        for j in range(len(file_names)):
            file_names[j]=new_folder_name+file_names[j]
        for file_name in file_names:
            file_list.append(file_name)
'''            
            true_value.append(i)
    print(file_list)
c=list(zip(file_list, true_value))
random.shuffle(c)
final_Path, y_train = zip(*c)
for file_path in final_Path:
    data=pd.read_csv(file_path,header=None)
    data=np.asarray(data)
    max_len=max(max_len,data.shape[0])
    print(max_len)

y_train=keras.utils.to_categorical(y_train, num_classes=134)
'''
x_train=[]
y_train=[]
for i, file_path in enumerate(file_list):
    data=pd.read_csv(file_path, header=None)
    data=np.asarray(data)
    data=normalize(data, axis=0)
    zero=np.zeros((1,8))
    for j in range(0, data.shape[0], 10):
        sample=data[j:min((j+10), data.shape[0]),:]
        while(sample.shape[0]<10):
            sample=np.insert(sample, 0, zero, axis=0)
        x_train.append(sample)
        y_train.append(int(file_path.split('/')[9])-1)   

y_train=np.asarray(y_train)
x_train=np.asarray(x_train)
y_train=keras.utils.to_categorical(y_train,  num_classes=134)
print(x_train.shape)
print(y_train.shape)

model = Sequential()
model.add(LSTM(256, activation='tanh', input_shape=(10, 8), kernel_initializer='orthogonal', return_sequences=True))
model.add(LSTM(128, activation='tanh', kernel_initializer='orthogonal', dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(134, activation='softmax'))

chk = ModelCheckpoint('lstm_best.h5', verbose=1, monitor='acc', save_best_only=True, mode="max")
#OPTIMIZER = SGD(lr=0.01)
OPTIMIZER = RMSProp(lr=0.1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=128, callbacks=[chk], epochs=1000, shuffle=True)
