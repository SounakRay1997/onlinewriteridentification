import numpy as np
import os
import pandas as pd
import csv
import math

path='/home/sounak/Desktop/BTP/Dataset/BIT_Database_New/english_modified/Test/'

window_size=11
G=1

def momentum_magnitude(neighbor,point):
    return point[4]*(math.sqrt(((neighbor[2]-point[2])/(neighbor[0]-point[0]))**2+((neighbor[3]-point[3])/(neighbor[0]-point[0]))**2))
    
def momentum_angle(neighbor,point):
    return np.arctan2((neighbor[3]-point[3]), (neighbor[2]-point[2]))
    
def stroke_fileno(val):
	return int(val.split('_')[0])

'''
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
    
def force_descriptor(window):
    # compute the Pattern of Local Gravitational Force representation
    # of the image, and then use the PLGF representation
    # to build the histogram of patterns
    #force_at_each_point=[]
    #forces=[]
    force=[0,0]
    centre_point=window[(len(window)-1)/2]
    window.remove(centre_point)
    for point in window:
        force=[a+b for a,b in zip(force,grav_force(point,centre_point))]
    force_mag=math.sqrt(force[1]**2+force[0]**2)
    force_angle=math.atan(force[1]/force[0])
    return force_mag,force_angle
    '''
    
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
        new_folder_name=folder_name+'Momentum_Magnitude_Angle/'
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
            fname=new_folder_name+stroke_files_copy[j].split('.')[0]+'_Momentum_Magnitude_Angle.csv'
            print(fname)
            with open(fname,'a') as csvfile:
                a=np.asarray([i for i in range(2)])
                writer=csv.writer(csvfile)
                writer.writerow(a)
            	for a in range(data.shape[0]-1):
            	    if(data[a][0]==data[(a+1)][0]):
            	        continue
            	    point_momentum_magnitude=momentum_magnitude(data[a+1],data[a])
            	    point_momentum_angle=momentum_angle(data[a+1],data[a])
                    print(a)  
                    momentum=np.asarray((point_momentum_magnitude, point_momentum_angle))
                    writer=csv.writer(csvfile)
                    if(math.isnan(momentum[1])==True or math.isnan(momentum[0])==True):
                        continue
                    writer.writerow(momentum)
            csvfile.close()     
