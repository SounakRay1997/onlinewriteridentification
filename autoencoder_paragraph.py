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
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

scaler=MinMaxScaler()
BATCH_SIZE=1
PATH="/home/sounak/Desktop/BTP/Dataset/BIT_Database_New/english_modified/Train/"
force_window_size=11
window_size=25
stride=5
number_of_bins=45
file_list=[]
true_value=[]
n_classes=15
max_len=0

def line_no(val):
	return val.split('_')[0]

for i in range(n_classes):
    paths=PATH+str(i+1)+'/'
    csv_files=[f for f in os.listdir(paths) if (f.endswith('.csv'))]
    for j in range(len(csv_files)):
        csv_files[j]=paths+csv_files[j]
    for files in csv_files:
        folder_name=files.split('.')[0]+'/'
        new_folder_name=folder_name+'PLGF_'+str(force_window_size)+'_window_size_per_point_corrected/Descriptor_Lines_'+str(window_size)+'_window_size_'+str(stride)+'_stride_'+str(number_of_bins)+'_number_of_bins/'
        file_names=[f for f in os.listdir(new_folder_name) if (f.endswith('.csv'))]
        file_names.sort(key=line_no)
        for j in range(len(file_names)):
            file_names[j]=new_folder_name+file_names[j]
        file_list.append(file_names)
        true_value.append(i)

x_train=[]
max_len=0
y_train=true_value
for paragraph in file_list:
    length=0
    for paragraph_line in paragraph:
        data=pd.read_csv(paragraph_line, header=None)
        data=np.asarray(data)
        length=length+data.shape[0]
    if(length>max_len):
        max_len=length
        
        
for paragraph in file_list:
    paragraph_data=np.empty(shape=(0,0))
    for i, paragraph_line in enumerate(paragraph):
        data=pd.read_csv(paragraph_line, header=None)
        data=np.asarray(data)
        if(i==0):
            paragraph_data=data
        else:
            paragraph_data=np.append(paragraph_data,data,axis=0)
    paragraph_data=preprocessing.scale(paragraph_data, axis=1)
    zero=np.zeros((1, (number_of_bins*2)))
    while(paragraph_data.shape[0]<max_len):
        paragraph_data=np.insert(paragraph_data, 0, zero, axis=0)
    x_train.append(paragraph_data)
    
    
x_train=np.asarray(x_train)
print(np.amax(x_train))
print(np.amin(x_train))

class lstm_bottleneck(tf.keras.layers.Layer):
    def __init__(self, lstm_units, time_steps, **kwargs):
        self.lstm_units = lstm_units
        self.time_steps = time_steps
        self.lstm_layer = tfkl.LSTM(lstm_units, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=2, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False)
        self.repeat_layer = tfkl.RepeatVector(time_steps)
        super(lstm_bottleneck, self).__init__(**kwargs)
    def call(self, inputs):
        return self.repeat_layer(self.lstm_layer(inputs))
    def compute_mask(self, inputs, mask=None):
        return mask

latent_dim=number_of_bins
time_steps = max_len
n_features = number_of_bins*2
input_layer = tfkl.Input(shape=(None, n_features))
x = tfk.layers.Masking(mask_value=0.0)(input_layer)
x = tfkl.Bidirectional(tfkl.LSTM(68, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', return_sequences=True), merge_mode='ave')(x)
x = lstm_bottleneck(lstm_units=latent_dim, time_steps=time_steps)(x)
x1 = tfkl.Bidirectional(tfkl.LSTM(68, activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', return_sequences=True), merge_mode='ave')(x)
x1 = tfkl.Bidirectional(tfkl.LSTM((number_of_bins*2), activation='tanh', recurrent_activation='sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', return_sequences=True), merge_mode='ave')(x1)
x1 = tfk.layers.Dense(n_features)(x1)
x2 = tfk.layers.Lambda(lambda x : x[:, 0, :])(x)
x2 = tfk.layers.Dense(n_classes, activation = 'softmax')(x2)
lstm_ae = tfk.models.Model(inputs=input_layer, outputs=[x2, x1])
lstm_ae.compile(optimizer='rmsprop', loss=['categorical_crossentropy', 'mse'])
print(lstm_ae.summary())

chk = ModelCheckpoint('/home/sounak/Desktop/BTP/CodeFiles/BIT_Database_New/paragraph_autoencoder.h5', verbose=1, monitor='val_loss', save_best_only=True, mode="min", save_weights_only=True)
early_stopping = EarlyStopping(monitor= "val_loss", patience=10, mode="min")

y_train=np.asarray(y_train)
y_train_svm=y_train
y_train=keras.utils.to_categorical(y_train,  num_classes=n_classes)

lstm_ae.fit(x_train, [y_train, x_train], batch_size=15, callbacks=[early_stopping, chk], epochs=1000, validation_split=0.1, shuffle=True)

intermediate_layer_model = tfk.models.Model(inputs=lstm_ae.input, outputs=lstm_ae.get_layer(index=3).output)
intermediate_layer_model.summary()
intermediate_output = intermediate_layer_model.predict(x_train)
print(intermediate_output[:, -1, :].shape)

param_grid = {'C': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000],  
              'gamma': [10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001], 
              'kernel': ['rbf', 'poly', 'sigmoid']} 
  
grid = GridSearchCV(SVC(), param_grid, refit = True, cv=2, verbose = 3) 
  
svm_input=intermediate_output[:, -1, :]

grid.fit(svm_input, y_train_svm) 
y_pred=grid.predict(svm_input)
print(y_pred)
print(y_train_svm)
acc=accuracy_score(y_train_svm, y_pred)
print("Train Accuracy :: ", acc)


PATH="/home/sounak/Desktop/BTP/Dataset/BIT_Database_New/english_modified/Test/"
test_file_list=[]
test_true_value=[]

for i in range(n_classes):
    paths=PATH+str(i+1)+'/'
    csv_files=[f for f in os.listdir(paths) if (f.endswith('.csv'))]
    for j in range(len(csv_files)):
        csv_files[j]=paths+csv_files[j]
    for files in csv_files:
        folder_name=files.split('.')[0]+'/'
        new_folder_name=folder_name+'PLGF_'+str(force_window_size)+'_window_size_per_point_corrected/Descriptor_Lines_'+str(window_size)+'_window_size_'+str(stride)+'_stride_'+str(number_of_bins)+'_number_of_bins/'
        file_names=[f for f in os.listdir(new_folder_name) if (f.endswith('.csv'))]
        file_names.sort(key=line_no)
        for j in range(len(file_names)):
            file_names[j]=new_folder_name+file_names[j]
        test_file_list.append(file_names)
        test_true_value.append(i)
        
x_test=[]
test_max_len=0
for paragraph in test_file_list:
    length=0
    for paragraph_line in paragraph:
        test_data=pd.read_csv(paragraph_line, header=None)
        test_data=np.asarray(test_data)
        length=length+test_data.shape[0]
    if(length>test_max_len):
        test_max_len=length
        
print(test_max_len)
        
for paragraph in test_file_list:
    paragraph_data=np.empty(shape=(0,0))
    for i, paragraph_line in enumerate(paragraph):
        data=pd.read_csv(paragraph_line, header=None)
        data=np.asarray(data)
        if(i==0):
            paragraph_data=data
        else:
            paragraph_data=np.append(paragraph_data,data,axis=0)
    paragraph_data=preprocessing.scale(paragraph_data, axis=1)
    zero=np.zeros((1, (number_of_bins*2)))
    while(paragraph_data.shape[0]<test_max_len):
        paragraph_data=np.insert(paragraph_data, 0, zero, axis=0)
    x_test.append(paragraph_data)
   
x_test=np.asarray(x_test)
    
intermediate_output = intermediate_layer_model.predict(x_test)
print(intermediate_output[:, -1, :].shape)

svm_test_input=intermediate_output[:, -1, :]
y_test=np.asarray(test_true_value)
y_pred=grid.predict(svm_test_input)
print(y_pred)
print(y_test)
acc=accuracy_score(y_test, y_pred)
print("Test Accuracy :: ", acc)
