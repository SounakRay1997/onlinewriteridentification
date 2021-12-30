import numpy as np
import os
import pandas as pd
import csv

path='/home/sounak/Desktop/BTP/Dataset/BIT_Database_New_LBP/chinese_english_modified/Train/'

substroke_size=9
step=4
#point=[]

def distance(neighbor,point):
    d=(neighbor[2]-point[2])**2+(neighbor[3]-point[3])**2
    return d

def stroke_fileno(val):
	return int(val.split('_')[0])

def descriptor(substroke):
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    encoding=[]
    
    for point in substroke:
        neighbors=[]
        distances=[]
        for neighbor in substroke:
            if neighbor!=point:
                neighbors.append(neighbor)
                distances.append(distance(neighbor,point))
        #neighbors.sort(key=distance)
        for i,neighbor in enumerate(neighbors):
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
    return encoding

for i in range(133):
    paths=path+str(i+1)+'/'
    csv_files=[f for f in os.listdir(paths) if (f.endswith('.csv'))]
    for j in range(len(csv_files)):
        csv_files[j]=paths+csv_files[j]
    for files in csv_files:
        line_number=1
        folder_name=files.split('.')[0]+'/'
        stroke_files=[f for f in os.listdir(folder_name) if (f.endswith('.csv'))] 
        stroke_files_copy=[]
        for j in range(len(stroke_files)):
            stroke_files_copy.append(stroke_files[j])
        new_folder_name=folder_name+'LBP_ss_'+str(substroke_size)+'_step_'+str(step)+'/'
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
            fname=new_folder_name+stroke_files_copy[j].split('.')[0]+str(substroke_size)+'_ss_LBP_'+str(step)+'_step.csv'
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
                	encoding=descriptor(substroke)
                	encoding=np.asarray(encoding)
                	writer=csv.writer(csvfile)
                	writer.writerow(encoding)
            csvfile.close()    
