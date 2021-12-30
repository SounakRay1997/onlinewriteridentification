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
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, normalize, MinMaxScaler
from keras.preprocessing import sequence
from keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam
import tensorflow as tf

scaler=MinMaxScaler()
BATCH_SIZE=1
PATH="/home/sounak/Desktop/BTP/Dataset/BIT_Database_New/english_modified/Train/"
force_window_size=11
window_size=25
stride=5
n_classes=30
number_of_bins=20

file_list=[]
true_value=[]
max_len=0
for i in range(n_classes):
    paths=PATH+str(i+1)+'/'
    csv_files=[f for f in os.listdir(paths) if (f.endswith('.csv'))]
    for j in range(len(csv_files)):
        csv_files[j]=paths+csv_files[j]
    for files in csv_files:
        folder_name=files.split('.')[0]+'/'
        new_folder_name=folder_name+'PLGF_'+str(force_window_size)+'_window_size_per_point_corrected/Descriptor_Lines_'+str(window_size)+'_window_size_'+str(stride)+'_stride_'+str(number_of_bins)+'_number_of_bins/'
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
min_len=max_len
for file_path in final_Paths:
    data=pd.read_csv(file_path,header=None)
    data=np.asarray(data)
    min_len=min(min_len, data.shape[0])
print(max_len)
print(min_len)

#y_train=keras.utils.to_categorical(y_train, num_classes=134)

x_train=[]

for i, file_path in enumerate(final_Paths):
    data=pd.read_csv(file_path,header=None)
    data=np.asarray(data)
    data=preprocessing.scale(data, axis=1)
    zero=np.zeros((1, (number_of_bins*2)))
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
        self.lstm_layer = tfkl.Bidirectional(tfkl.LSTM(lstm_units, recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', return_sequences=False), merge_mode='ave')
        self.repeat_layer = tfkl.RepeatVector(time_steps)
        super(lstm_bottleneck, self).__init__(**kwargs)
    def call(self, inputs):
        return self.repeat_layer(self.lstm_layer(inputs))
    def compute_mask(self, inputs, mask=None):
        return mask

latent_dim=30
time_steps = max_len
n_features = number_of_bins*2
input_layer = tfkl.Input(shape=(None, n_features))
x = tfk.layers.Masking(mask_value=0.0)(input_layer)
#x = tfkl.Bidirectional(tfkl.LSTM(37, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', return_sequences=True), merge_mode='ave')(x)
x = tfkl.Bidirectional(tfkl.LSTM(35, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', return_sequences=True), merge_mode='ave')(x)
x = lstm_bottleneck(lstm_units=latent_dim, time_steps=time_steps)(x)
x1 = tfkl.Bidirectional(tfkl.LSTM(35, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', return_sequences=True), merge_mode='ave')(x)
#x1 = tfkl.Bidirectional(tfkl.LSTM(37, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', return_sequences=True), merge_mode='ave')(x1)
x1 = tfkl.Bidirectional(tfkl.LSTM(40, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', return_sequences=True), merge_mode='ave')(x1)
x1 = tfk.layers.Dense(n_features)(x1)
x2 = tfk.layers.Lambda(lambda x : x[:, 0, :])(x)
x2 = tfk.layers.Dense(n_classes, activation = 'softmax')(x2)
lstm_ae = tfk.models.Model(inputs=input_layer, outputs=[x2, x1])
lstm_ae.compile(optimizer='rmsprop', loss=['categorical_crossentropy', 'mse'])
print(lstm_ae.summary())

chk = ModelCheckpoint('/home/sounak/Desktop/BTP/CodeFiles/BIT_Database_New/supervised_autoencoder_bidirectional_best_40_features_30_classes.h5', verbose=1, monitor='val_loss', save_best_only=True, mode="min", save_weights_only=True)
reduce_lr=ReduceLROnPlateau(monitor='dense_loss', factor=0.2, patience=5, min_lr=0.00000001)

y_train=np.asarray(y_train)
y_train=keras.utils.to_categorical(y_train,  num_classes=n_classes)

lstm_ae.fit(x_train, [y_train, x_train], batch_size=15, callbacks=[reduce_lr, chk], epochs=1000, validation_split=0.1, shuffle=True)

intermediate_layer_model = tfk.models.Model(inputs=lstm_ae.input, outputs=lstm_ae.get_layer(index=3).output)
intermediate_layer_model.summary()
intermediate_output = intermediate_layer_model.predict(x_train)
print(intermediate_output[:, -1, :].shape)
svm_input=intermediate_output[:, -1, :]

param_grid = {'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],  
              'gamma': [10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001], 
              'kernel': ['rbf', 'poly', 'sigmoid']}  
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3, cv=10) 
  
grid.fit(svm_input, y_train) 
y_pred=grid.predict(svm_input)
print(y_pred)
print(y_train)
acc=accuracy_score(y_train, y_pred)
print("Train Accuracy :: ", acc)

PATH="/home/sounak/Desktop/BTP/Dataset/BIT_Database_New/english_modified/Test/"
force_window_size=11
window_size=25
stride=5
test_file_list=[]
test_true_value=[]

for i in range(50):
    paths=PATH+str(i+1)+'/'
    csv_files=[f for f in os.listdir(paths) if (f.endswith('.csv'))]
    for j in range(len(csv_files)):
        csv_files[j]=paths+csv_files[j]
    for files in csv_files:
        folder_name=files.split('.')[0]+'/'
        new_folder_name=folder_name+'PLGF_'+str(force_window_size)+'_window_size_per_point_corrected/Descriptor_Lines_'+str(window_size)+'_window_size_'+str(stride)+'_stride_'+str(number_of_bins)+'_number_of_bins/'
        file_names=[f for f in os.listdir(new_folder_name) if (f.endswith('.csv'))]
        for j in range(len(file_names)):
            file_names[j]=new_folder_name+file_names[j]
        for file_name in file_names:
            test_file_list.append(file_name)
            test_true_value.append(i)
print(len(test_file_list))
y_test=test_true_value
max_len=0
for file_path in test_file_list:
    data=pd.read_csv(file_path,header=None)
    data=np.asarray(data)
    max_len=max(max_len, data.shape[0])
    
x_test=[]

for i, file_path in enumerate(test_file_list):
    data=pd.read_csv(file_path,header=None)
    data=np.asarray(data)
    data=preprocessing.scale(data, axis=1)
    zero=np.zeros((1, (number_of_bins*2)))
    while(data.shape[0]<max_len):
        data=np.insert(data, 0, zero, axis=0)
    x_test.append(data)
    
y_test=np.asarray(y_test)
x_test=np.asarray(x_test)

test_intermediate_output = intermediate_layer_model.predict(x_test)
print(test_intermediate_output[:, -1, :].shape)
test_svm_input=test_intermediate_output[:, -1, :]

y_pred=grid.predict(test_svm_input)
print(y_pred)
print(y_test)
acc=accuracy_score(y_test, y_pred)
print("Test Accuracy :: ", acc)

