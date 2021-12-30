import pandas as pd
import numpy as np
import os
import csv
import math

def euclidean_Distance(x1, y1, x2, y2):
    return math.sqrt(pow((x2-x1), 2)+pow((y2-y1), 2))
    
    
path='/home/sounak/Desktop/BTP/Dataset/BIT_Database_New_PLGF/chinese_english_modified/Train/'

def stroke_fileno(val):
	return int(val.split('_')[0])

for i in range(133):
    paths=path+str(i+1)+'/'
    csv_files=[f for f in os.listdir(paths) if (f.endswith('.csv'))]
    for j in range(len(csv_files)):
        csv_files[j]=paths+csv_files[j]
    for files in csv_files:
        line_number=1
        folder_name=files.split('.')[0]+'/'
        stroke_files=[f for f in os.listdir(folder_name) if (f.endswith('.csv'))] 
        stroke_files.sort(key=stroke_fileno)
        for j in range(len(stroke_files)):
        	stroke_files[j]=folder_name+stroke_files[j]
        print(stroke_files)
        for j in range(len(stroke_files)-1):
            data=pd.read_csv(stroke_files[j])
            data_next=pd.read_csv(stroke_files[(j+1)])
            data=np.asarray(data)
            data_next=np.asarray(data_next)
            last_element=np.asarray((data[(data.shape)[0]-1][2], data[(data.shape)[0]-1][3]))
            first_element=np.asarray((data_next[0][2], data_next[0][3]))
            dist=euclidean_Distance(last_element[0], last_element[1], first_element[0], first_element[1])
            os.rename(stroke_files[j], (stroke_files[j].split('.')[0]+'_line_'+str(line_number)+'.csv'))
            if(dist>10000):
                line_number=line_number+1
        os.rename(stroke_files[len(stroke_files)-1], (stroke_files[len(stroke_files)-1].split('.')[0]+'_line_'+str(line_number)+'.csv'))
            
            
            
            
            
    
