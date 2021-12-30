import os
import random
import keras
import pickle 
import math
import pandas as pd

from sklearn.preprocessing import StandardScaler
from keras.utils import Sequence
from keras.models import Model
from keras.preprocessing import image
import numpy as np

from keras.layers import Input
from keras import backend as K
from keras.models import Sequential

import tensorflow as tf
from keras.backend.common import _EPSILON
window_size=9
substroke_size=15

class DataGenerator(Sequence):
    def __init__(self, BATCH_SIZE, PATH):
        self.file_list=[]
        self.true_value=[]
        self.min_len=10000
        self.BATCH_SIZE=BATCH_SIZE
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
                    self.file_list.append(file_name)
                    self.true_value.append(i)
        c=list(zip(self.file_list, self.true_value))
        random.shuffle(c)
        self.final_Path, self.final_true_value = zip(*c)
        for file_path in self.final_Path:
            data=pd.read_csv(file_path,header=None)
            data=np.asarray(data)
            self.min_len=min(self.min_len, data.shape[0])
        
    def __len__(self):
        return math.ceil(len(self.final_true_value)/self.BATCH_SIZE)
        
    def __getitem__(self, idx):
        batch_paths = self.final_Path[idx * self.BATCH_SIZE:(idx + 1) * self.BATCH_SIZE]
        batch_true_value = self.final_true_value[idx * self.BATCH_SIZE:(idx + 1) * self.BATCH_SIZE]  
        batch_true_value=keras.utils.to_categorical(batch_true_value, num_classes=134)
        
        max_len_val=0
        batch_data=np.zeros((len(batch_paths), self.min_len, 8), dtype=np.float32)
        for file_path in batch_paths:
            data=pd.read_csv(file_path,header=None)
            data=np.asarray(data)
            max_len_val=max(max_len_val, data.shape[0])
        for i, file_path in enumerate(batch_paths):
            data=pd.read_csv(file_path,header=None)
            data=np.asarray(data)
            #data=data[:, [0, 1, 2, 3, 6, 7]]
            scaler=StandardScaler()
            scaler.fit(data)
            data=scaler.transform(data)
            #zero=np.zeros((1,6))
            #while(data.shape[0]<self.max_len):
            #    data=np.append(data, zero, axis=0)
            data=data[:self.min_len, :]
            batch_data[i]=data
        return batch_data, batch_true_value
             
            
