import numpy as np
import os
import pandas as pd
import csv
import math
from sklearn import preprocessing as pre

path='/home/sounak/Desktop/BTP/Dataset/BIT_Database_New/english_modified/Train/'

force_window_size=11
window_size=40
stride=25

def encode(PLGF_array):
    point_descriptor=[]
    force=list(PLGF_array[:, 0])
    centre_point=force[(window_size-1)/2]
    force.remove(centre_point)
    difference=[]
    for val in force:
        difference.append((val-centre_point))
    positive_sum=0.0
    negative_sum=0.0
    positive_count=0
    negative_count=0
    for value in difference:
        if(value>=0):
            positive_sum=positive_sum+value
            positive_count=positive_count+1
        else:
            negative_sum=negative_sum+value
            negative_count=negative_count+1
    positive_mean=0
    negative_mean=0
    if(positive_count>0):
        positive_mean=positive_sum/positive_count
    if(negative_count>0):
        negative_mean=negative_sum/negative_count
    force_descriptor=[0, 0, 0, 0]
    for value in difference:
        if(value>=0):
            if(value>=positive_mean):
                force_descriptor[3]=force_descriptor[3]+1
            else:
                force_descriptor[2]=force_descriptor[2]+1
        if(value<0):
            if(value>=negative_mean):
                force_descriptor[1]=force_descriptor[1]+1
            else:
                force_descriptor[0]=force_descriptor[0]+1
    for value in force_descriptor:
        point_descriptor.append(value)
    mag_array=PLGF_array[:, 0]
    #mean=np.mean(mag_array)
    #std=np.std(mag_array)
    #point_descriptor.append(mean)
    #point_descriptor.append(std)
    angle=list(PLGF_array[:, 1])
    centre_point_angle=angle[(window_size-1)/2]
    angle.remove(centre_point_angle)
    threshold=(math.pi)/8
    difference_angle=[]
    for val in angle:
        difference_angle.append(abs(val-centre_point_angle))
    angle_descriptor=[0, 0]
    for value in difference_angle:
        if(value>=threshold):
            angle_descriptor[1]=angle_descriptor[1]+1
        else:
            angle_descriptor[0]=angle_descriptor[0]+1
    for value in angle_descriptor:
        point_descriptor.append(value)
    point_descriptor=np.asarray(point_descriptor)
    return point_descriptor
        
         

for i in range(134):
    paths=path+str(i+1)+'/'
    csv_files=[f for f in os.listdir(paths) if (f.endswith('.csv'))]
    for j in range(len(csv_files)):
        csv_files[j]=paths+csv_files[j]
    for files in csv_files:
        folder_name=files.split('.')[0]+'/'
        new_folder_name=folder_name+'PLGF_'+str(force_window_size)+'_window_size_per_point/'
        print(new_folder_name)
        plgf_files=[f for f in os.listdir(new_folder_name) if (f.endswith('.csv'))]
        plgf_files_full_path=[]
        for plgf_file in plgf_files:
            plgf_files_full_path.append((new_folder_name+plgf_file))
        plgf_folder=new_folder_name+'PLGF_Descriptor_Lines_'+str(window_size)+'_window_size_'+str(stride)+'_stride/'
        os.mkdir(plgf_folder)
        for file_path, file_name in zip(plgf_files_full_path,plgf_files):
            fname=plgf_folder+'Line'+file_name.split('_')[3]+'_PLGF_'+str(window_size)+'_window_size_'+str(stride)+'_stride.csv'
            print(file_path)
            print(file_name)
            data=pd.read_csv(file_path)
            data=np.asarray(data)
            descriptors_of_a_point=[]
            with open(fname,'a') as csvfile:
                writer=csv.writer(csvfile)
                for a in range(0, data.shape[0], stride):
            	    if((a-((window_size-1)/2))<0):
            	        points=data[0:(a+1+((window_size-1)/2)),:]
            	        for i in range(abs(a-((window_size-1)/2))):
            	            b=np.asarray((0, 0))
            	            b=np.reshape(b, (1, 2))
            	            points=np.insert(points, 0, b, axis=0) 
            	    elif((a+((window_size-1)/2))>(data.shape[0]-1)):
            	        points=data[(a-((window_size-1)/2)):data.shape[0],:]
            	        for i in range(abs((data.shape[0])-(a+((window_size-1)/2)))+1):
            	            b=np.asarray((0, 0))
            	            b=np.reshape(b, (1, 2))
            	            points=np.concatenate((points, b), axis=0)
            	    else:
            	        points=data[(a-((window_size-1)/2)):(a+1+((window_size-1)/2)),:]  
                    print(points)                
                    descriptor_val=encode(points)
                    descriptors_of_a_point.append(descriptor_val)
                for val in descriptors_of_a_point:
                    writer.writerow(val)
            csvfile.close()
            
