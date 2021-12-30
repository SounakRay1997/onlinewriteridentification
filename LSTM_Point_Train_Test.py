import os
import random
import keras
import pickle 
import math
import pandas as pd
import numpy as np
from batch_generators import DataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler,normalize
from keras.layers import LSTM, Bidirectional, Dropout, Input
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense

BATCH_SIZE=128
PATH="/home/sounak/Desktop/BTP/Dataset/BIT_Database_New/chinese_english_modified/Train/"
TEST_PATH="/home/sounak/Desktop/BTP/Dataset/BIT_Database_New/chinese_english_modified/Test/"
window_size=11

file_list=[]
true_value=[]
for i in range(25):
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
            
test_file_list=[]
test_true_value=[]
for i in range(25):
    test_paths=TEST_PATH+str(i+1)+'/'
    test_csv_files=[f for f in os.listdir(test_paths) if (f.endswith('.csv'))]
    for j in range(len(test_csv_files)):
        test_csv_files[j]=test_paths+test_csv_files[j]
    for files in test_csv_files:
        folder_name=files.split('.')[0]+'/'
        new_folder_name=folder_name+'PLGF_'+str(window_size)+'_window_size_per_point/PLGF_Descriptor_Lines_'+str(window_size)+'_per_point/'
        test_file_names=[f for f in os.listdir(new_folder_name) if (f.endswith('.csv'))]
        for j in range(len(test_file_names)):
            test_file_names[j]=new_folder_name+test_file_names[j]
        for file_name in test_file_names:
            test_file_list.append(file_name)            
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
    #data=normalize(data, axis=0)
    zero=np.zeros((1,6))
    j=0
    while(j<data.shape[0]):
        sample=data[j:min((j+50), data.shape[0]),:]
        while(sample.shape[0]<50):
            sample=np.insert(sample, 0, zero, axis=0)
        x_train.append(sample)
        y_train.append(int(file_path.split('/')[9])-1) 
        step = random.randint(1, 12)
        j += step  

y_train=np.asarray(y_train)
x_train=np.asarray(x_train)
y_train=keras.utils.to_categorical(y_train,  num_classes=25)
print(x_train.shape)
print(y_train.shape)

x_test=[]
y_test=[]
for i, file_path in enumerate(test_file_list):
    data=pd.read_csv(file_path, header=None)
    data=np.asarray(data)
    #data=normalize(data, axis=0)
    zero=np.zeros((1,6))
    j=0
    while(j<data.shape[0]):
        sample=data[j:min((j+50), data.shape[0]),:]
        while(sample.shape[0]<50):
            sample=np.insert(sample, 0, zero, axis=0)
        x_test.append(sample)
        y_test.append(int(file_path.split('/')[9])-1) 
        step = random.randint(1, 12)
        j += step  

y_test=np.asarray(y_test)
x_test=np.asarray(x_test)
y_test=keras.utils.to_categorical(y_test,  num_classes=25)
print(x_test.shape)
print(y_test.shape)

model = Sequential()
model.add(LSTM(64, activation='tanh', return_sequences=True, dropout=0.20, recurrent_dropout=0.20, input_shape=(50, 6)))
model.add(LSTM(128, activation='tanh', dropout=0.20, recurrent_dropout=0.20))
model.add(Dense(25, activation='softmax'))

chk = ModelCheckpoint('chinese_english_lstm_best_25_classes_greater_parameter_64_128_greater_data_input_shape_50_with_dropout_15_percent_1500_epochs_relu_without_normalize.h5', verbose=1, monitor='val_loss', save_best_only=True, mode="min")
#OPTIMIZER = SGD(lr=0.01)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=15, min_lr=1e-10, verbose=1, mode="min")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, validation_data=(x_test,y_test), batch_size=64, callbacks=[chk, reduce_lr], epochs=1500, shuffle=True)
