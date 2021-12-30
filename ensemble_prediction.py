import os
import random
import keras
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score
import pandas as pd
import csv


TEST_PATH="/home/sounak/Desktop/BTP/Dataset/BIT_Database_New/chinese_english_modified/Test/"
window_size=11

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
        y_test.append(file_path.split('/')[10]) 
        step = random.randint(1, 20)
        j += step  
        
        
model=load_model('chinese_english_lstm_best_25_classes_greater_parameter_64_128_greater_data_input_shape_50_with_dropout_15_percent_1500_epochs_relu_without_normalize.h5.h5')
x_test=np.asarray(x_test)     
y_pred=model.predict(x_test)

sum_val=np.zeros((1, 25))
mean=0
count=0
y_pred_ensemble=[]

for i in range(y_pred.shape[0]):
    if(y_test[i]==y_test[min((y_pred.shape[0]-1), i+1)] or i==(y_pred.shape[0]-1)):
        a=np.reshape(y_pred[i], (1, 25))
        sum_val=sum_val+a
        count=count+1
    else:
        a=np.reshape(y_pred[i], (1, 25))
        sum_val=sum_val+a
        count=count+1
        mean=sum_val/count
        y_pred_ensemble.append(mean)
        sum_val=np.zeros((1, 25))
        count=0


mean=sum_val/count
y_pred_ensemble.append(mean)    
y_pred_ensemble=np.asarray(y_pred_ensemble)
print(y_pred_ensemble)

y_pred=np.argmax(y_pred_ensemble, axis=-1)
print(y_pred)
y_true=[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24]

print ("Test Accuracy :: ", accuracy_score(y_true, y_pred))

