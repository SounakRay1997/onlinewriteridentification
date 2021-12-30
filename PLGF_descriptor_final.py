import numpy as np
import os
import pandas as pd
import csv
import math

path='/home/sounak/Desktop/BTP/Dataset/BIT_Database_New/english_modified/Test/'

window_size=11
G=1

def distance(neighbor,point):
    return math.sqrt((neighbor[2]-point[2])**2+(neighbor[3]-point[3])**2)
  
    
    
    
def grav_force(neighbor, point):
    force_mag=(G*neighbor[4]*point[4])/(distance(neighbor, point)**2)
    unit_vec=[(neighbor[2]-point[2]),(neighbor[3]-point[3])]
    unit_vec[0]=unit_vec[0]/distance(neighbor,point)
    unit_vec[1]=unit_vec[1]/distance(neighbor,point)
    force_vec=[0,0]
    force_vec[0]=force_mag*unit_vec[0]
    force_vec[1]=force_mag*unit_vec[1]
    return force_vec   
    
    
def stroke_fileno(val):
	return int(val.split('_')[0])


def force_descriptor(window):
    force=[0,0]
    centre_point=window[(len(window)-1)/2]
    window.remove(centre_point)
    for point in window:
        force=[a+b for a,b in zip(force,grav_force(point,centre_point))]
    force_mag=math.sqrt(force[1]**2+force[0]**2)
    force_angle=np.arctan2(force[1], force[0])
    return force_mag,force_angle
    
    
for i in range(134):
    paths=path+str(i+1)+'/'
    csv_files=[f for f in os.listdir(paths) if (f.endswith('.csv'))]
    for j in range(len(csv_files)):
        csv_files[j]=paths+csv_files[j]
    for files in csv_files:
        folder_name=files.split('.')[0]+'/'
        stroke_files=[f for f in os.listdir(folder_name) if (f.endswith('_processed.csv'))] 
        stroke_files_copy=[]
        for j in range(len(stroke_files)):
            stroke_files_copy.append(stroke_files[j])
        new_folder_name=folder_name+'PLGF_'+str(window_size)+'_window_size_per_point_corrected/'
        os.mkdir(new_folder_name)
        #stroke_files.sort(key=stroke_fileno)
        for j in range(len(stroke_files)):
        	stroke_files[j]=folder_name+stroke_files[j]
        print(stroke_files)
        for j in range(len(stroke_files)):
            print(stroke_files[j])
            data=pd.read_csv(stroke_files[j])
            data=np.asarray(data)
            print(stroke_files_copy[j])
            fname=new_folder_name+stroke_files_copy[j].split('.')[0]+'_PLGF_'+str(window_size)+'_window_size.csv'
            print(fname)
            with open(fname,'a') as csvfile:
                a=np.asarray([i for i in range(2)])
                writer=csv.writer(csvfile)
                writer.writerow(a)
            	for a in range(data.shape[0]):
            	    if((a-((window_size-1)/2))<0):
            	        window_vals=data[0:(a+1+((window_size-1)/2)),:]
            	        for i in range(abs(a-((window_size-1)/2))):
            	            b=np.asarray((0, 0, 0, 0, 0))
            	            b=np.reshape(b, (1, 5))
            	            window_vals=np.insert(window_vals, 0, b, axis=0) 
            	    elif((a+((window_size-1)/2))>(data.shape[0]-1)):
            	        window_vals=data[(a-((window_size-1)/2)):data.shape[0],:]
            	        for i in range(abs((data.shape[0])-(a+((window_size-1)/2)))+1):
            	            b=np.asarray((0, 0, 0, 0, 0))
            	            b=np.reshape(b, (1, 5))
            	            window_vals=np.concatenate((window_vals, b), axis=0)
            	    else:
            	        window_vals=data[(a-((window_size-1)/2)):(a+1+((window_size-1)/2)),:]  
                    print(window_vals)
                    print(a)  
                    window_vals=list(window_vals)
                    for l,point in enumerate(window_vals):
                    	window_vals[l]=list(point)
                    encoding=force_descriptor(window_vals)
                    encoding=np.asarray(encoding)
                    writer=csv.writer(csvfile)
                    if(math.isnan(encoding[1])==True or math.isnan(encoding[0])==True):
                        continue
                    writer.writerow(encoding)
            csvfile.close()     
