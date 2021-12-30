import numpy as np
import os
import pandas as pd
import csv

path='/home/sounak/Desktop/BTP/Dataset/BIT_Database_New/english_modified/Test/'

substroke_size=9
G=1
step=4
#point=[]

def distance(neighbor,point):
    return ((neighbor[2]-point[2])**2+(neighbor[3]-point[3])**2)
    
def grav_force(neighbor, point):
    return (G*neighbor[4]*point[4])/distance(neighbor, point)    
    
def stroke_fileno(val):
	return int(val.split('_')[0])

def force_descriptor(substroke):
    # compute the Pattern of Local Gravitational Force representation
    # of the image, and then use the PLGF representation
    # to build the histogram of patterns
    force_at_each_point=[]
    for point in substroke:
        neighbors=[]
        distances=[]
        for neighbor in substroke:
            if neighbor!=point:
                neighbors.append(neighbor)
                distances.append(distance(neighbor,point))
        s=0
        for i,neighbor in enumerate(neighbors):
            neighbors[i]=grav_force(neighbor,point)
            s+=neighbors[i]
        force_at_each_point.append(s)   
        '''for i,neighbor in enumerate(neighbors):
            if(neighbor[4]>point[4]):
                neighbors[i]=1
            else:
                neighbors[i]=0
        Z=[x for _, x in sorted(zip(distances, neighbors),reverse=True)]
        i=0
        p=0
        for neighbor in Z:
            if(neighbor):
                p+=2**i
            i+=1
        encoding.append(p)    
    for force in force_at_each_point:'''
    return force_at_each_point
    
for i in range(134):
    paths=path+str(i+1)+'/'
    csv_files=[f for f in os.listdir(paths) if (f.endswith('.csv'))]
    for j in range(len(csv_files)):
        csv_files[j]=paths+csv_files[j]
    for files in csv_files:
        line_number=1
        folder_name=files.split('.')[0]+'/'
        stroke_files=[f for f in os.listdir(folder_name) if (f.endswith('_processed.csv'))] 
        stroke_files_copy=[]
        for j in range(len(stroke_files)):
            stroke_files_copy.append(stroke_files[j])
        new_folder_name=folder_name+'PLGF_ss_'+str(substroke_size)+'_step_'+str(step)+'/'
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
            fname=new_folder_name+stroke_files_copy[j].split('.')[0]+'_'+str(substroke_size)+'_ss_PLGF_'+str(step)+'_step.csv'
            print(fname)
            with open(fname,'a') as csvfile:
                a=np.asarray([i for i in range(substroke_size)])
                writer=csv.writer(csvfile)
                writer.writerow(a)
            	for a in range(0,(data.shape[0]),step):
                	substroke=data[a:min(a+substroke_size,data.shape[0]),:]
                	substroke=list(substroke)
                	for l,point in enumerate(substroke):
                		substroke[l]=list(point)
                	encoding=force_descriptor(substroke)
                	encoding=np.asarray(encoding)
                	writer=csv.writer(csvfile)
                	writer.writerow(encoding)
            csvfile.close()     
