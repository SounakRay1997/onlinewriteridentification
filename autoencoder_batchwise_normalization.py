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
from keras.layers import RepeatVector, Masking
from keras.layers import TimeDistributed
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import Sequence

EPOCHS=1000
BATCH_SIZE=10
PATH="/home/sounak/Desktop/BTP/Dataset/BIT_Database_New/english_modified/Train/"
window_size=11
substroke_size=11
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
        new_folder_name=folder_name+'PLGF_'+str(window_size)+'_window_size_per_point/PLGF_Descriptor_Lines_'+str(substroke_size)+'_per_point/'
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

'''#y_train=keras.utils.to_categorical(y_train, num_classes=134)

x_train=[]

for i, file_path in enumerate(final_Paths):
    data=pd.read_csv(file_path,header=None)
    data=np.asarray(data)
    data=normalize(data, axis=0)
    zero=np.zeros((1, 6))
    while(data.shape[0]<max_len):
        data=np.insert(data, 0, zero, axis=0)
    x_train.append(data)'''
       
class DataGenerator(Sequence):

    def __init__(self,path):
        self.paths = []
        for i in range(134):
            paths=path+str(i+1)+'/'
            csv_files=[f for f in os.listdir(paths) if (f.endswith('.csv'))]
            for j in range(len(csv_files)):
                csv_files[j]=paths+csv_files[j]
            for files in csv_files:
                folder_name=files.split('.')[0]+'/'
                new_folder_name=folder_name+'PLGF_'+str(window_size)+'_window_size_per_point/PLGF_Descriptor_Lines_'+str(substroke_size)+'_per_point/'
                file_names=[f for f in os.listdir(new_folder_name) if (f.endswith('.csv'))]
                for j in range(len(file_names)):
                    file_names[j]=new_folder_name+file_names[j]
                for file_name in file_names:
                    self.paths.append(file_name)
        
    def __len__(self):
        return math.ceil(len(self.paths) / BATCH_SIZE)

    def __getitem__(self, idx):
        batch_paths = self.paths[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE]
        batch_descriptors=[]
        for i, file_path in enumerate(batch_paths):
            data=pd.read_csv(file_path,header=None)
            data=np.asarray(data)
            data=normalize(data, axis=0)
            zero=np.zeros((1, 6))
            while(data.shape[0]<max_len):
                data=np.insert(data, 0, zero, axis=0)
            batch_descriptors.append(data)
            
        batch_descriptors=np.asarray(batch_descriptors)   
        return batch_descriptors, batch_descriptors

#x_train=np.asarray(x_train)
#print(x_train.shape)
 
model = Sequential()  
model.add(Masking(mask_value=0.0, input_shape=(max_len, 6)))
model.add(LSTM(16, activation='relu', return_sequences=True))
model.add(LSTM(12, activation='relu', return_sequences=True))
model.add(LSTM(8, activation='tanh', return_sequences=False))
model.add(RepeatVector(max_len))

model.add(LSTM(16, activation='relu', return_sequences=True))
model.add(LSTM(12, activation='relu', return_sequences=True))
model.add(LSTM(6, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(6, activation = 'relu')))

chk = ModelCheckpoint('autoencoder_best_batchwise_normalization_batch_size_10_new.h5', verbose=1, monitor='loss', save_best_only=True, mode="min")
#OPTIMIZER = SGD(lr=0.01)
model.compile(loss='mse', optimizer='adam')
model.summary()

train_datagen = DataGenerator(PATH)
model.fit_generator(generator=train_datagen, epochs=EPOCHS, callbacks=[chk])
#model.fit(x_train, x_train, batch_size=64, callbacks=[chk], epochs=1000, validation_split=0.1, shuffle=True)
