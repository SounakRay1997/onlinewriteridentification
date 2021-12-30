import os
import random
import keras
import pickle 
import math
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from keras.models import load_model, Model
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

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
    
x_train=[]

for i, file_path in enumerate(final_Paths):
    data=pd.read_csv(file_path,header=None)
    data=np.asarray(data)
    #data=normalize(data, axis=0)
    zero=np.zeros((1, 6))
    while(data.shape[0]<max_len):
        data=np.insert(data, 0, zero, axis=0)
    x_train.append(data)
    
y_train=np.asarray(y_train)
x_train=np.asarray(x_train)

model=load_model('autoencoder_best_seq_len_bs_20.h5')
w=model.get_weights()
print(w)
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(index=4).output)
intermediate_output = intermediate_layer_model.predict(x_train)[-1]

clf = SVC(gamma='auto')
clf.fit(intermediate_output, y_train)
y_pred=clf.predict(intermediate_output)
print(y_pred)
print(y_train)
acc=accuracy_score(y_train, y_pred)
print(acc)
