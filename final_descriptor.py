import numpy as np
import os
import pandas as pd
import csv
import math
from sklearn import preprocessing as pre
import shutil

path='/home/sounak/Desktop/BTP/Dataset/BIT_Database_New/english_modified/Train/'

force_window_size=11
window_size=25
stride=5
number_of_bins=20

def plgf_encode(PLGF_array):
    plgf_histogram=np.zeros((1,number_of_bins))
    for i in range(PLGF_array.shape[0]):
        for j in range(number_of_bins):
            if(PLGF_array[i][1]>(-math.pi+(j*((2*math.pi)/number_of_bins))) and PLGF_array[i][1]<=(-math.pi+((j+1)*((2*math.pi)/number_of_bins)))):
                plgf_histogram[0][j]=plgf_histogram[0][j]+PLGF_array[i][0]
                break
    return plgf_histogram

def momentum_encode(Momentum_array):
    momentum_histogram=np.zeros((1,number_of_bins))
    for i in range(Momentum_array.shape[0]):
        for j in range(number_of_bins):
            if(Momentum_array[i][1]>(-math.pi+(j*((2*math.pi)/number_of_bins))) and Momentum_array[i][1]<=(-math.pi+((j+1)*((2*math.pi)/number_of_bins)))):
                momentum_histogram[0][j]=momentum_histogram[0][j]+Momentum_array[i][0]
                break
    return momentum_histogram
    
def stroke_fileno(val):
	return int(val.split('_')[0])

for i in range(134):
    paths=path+str(i+1)+'/'
    csv_files=[f for f in os.listdir(paths) if (f.endswith('.csv'))]
    for j in range(len(csv_files)):
        csv_files[j]=paths+csv_files[j]
    for files in csv_files:
        folder_name=files.split('.')[0]+'/'
        new_folder_name=folder_name+'PLGF_'+str(force_window_size)+'_window_size_per_point_corrected/'
        momentum_folder_name = folder_name+'Momentum_Magnitude_Angle/'
        plgf_files=[f for f in os.listdir(new_folder_name) if (f.endswith('.csv'))]
        momentum_files=[f for f in os.listdir(momentum_folder_name) if (f.endswith('.csv'))]
        plgf_files.sort(key=stroke_fileno)
        momentum_files.sort(key=stroke_fileno)
        plgf_files_full_path=[]
        momentum_files_full_path=[]
        for plgf_file in plgf_files:
            plgf_files_full_path.append((new_folder_name+plgf_file))
        for momentum_file in momentum_files:
            momentum_files_full_path.append((momentum_folder_name+momentum_file))
        descriptor_folder=new_folder_name+'Descriptor_Lines_'+str(window_size)+'_window_size_'+str(stride)+'_stride_'+str(number_of_bins)+'_number_of_bins/'
        os.mkdir(descriptor_folder)
        for plgf_file_path, momentum_file_path, file_name in zip(plgf_files_full_path, momentum_files_full_path, plgf_files):
            fname=descriptor_folder+'Line'+file_name.split('_')[3]+'_Descriptor_'+str(window_size)+'_window_size_'+str(stride)+'_stride_'+str(number_of_bins)+'_number_of_bins.csv'
            print(plgf_file_path)
            print(momentum_file_path)
            data_plgf=pd.read_csv(plgf_file_path)
            data_plgf=np.asarray(data_plgf)
            print(data_plgf.shape)
            data_momentum=pd.read_csv(momentum_file_path)
            data_momentum=np.asarray(data_momentum)
            print(data_momentum.shape)
            final_descriptor=[]
            with open(fname,'a') as csvfile:
                writer=csv.writer(csvfile)
                for a in range(0, data_momentum.shape[0], stride):
                    plgf_window=data_plgf[a:(a+window_size),:] 
                    momentum_window=data_momentum[a:(a+window_size):,:]                                
                    plgf_descriptor_val=plgf_encode(plgf_window)
                    momentum_descriptor_val=momentum_encode(momentum_window)
                    substroke_descriptor=np.append(plgf_descriptor_val, momentum_descriptor_val, axis=1)
                    substroke_descriptor=list(substroke_descriptor[0])
                    print(substroke_descriptor)
                    final_descriptor.append(substroke_descriptor)
                    if((a+window_size)>min(data_momentum.shape[0], data_plgf.shape[0])):
                        break
                for val in final_descriptor:
                    writer.writerow(val)
            csvfile.close()
