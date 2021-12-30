import numpy as np
import csv
import os

path='/home/sounak/Desktop/BTP/Dataset/BIT_Database_New/english_modified/Train/'

for i in range(134):
    paths=path+str(i+1)+'/'
    csv_files=[f for f in os.listdir(paths) if (f.endswith('.csv'))]
    for j in range(len(csv_files)):
        csv_files[j]=paths+csv_files[j]
    for files in csv_files:
        line_number=1
        folder_name=files.split('.')[0]+'/'
        stroke_files=[f for f in os.listdir(folder_name) if (f.endswith('_ss_9_step_4.csv'))]
        for j in range(len(stroke_files)):
        	stroke_files[j]=folder_name+stroke_files[j]
        print(stroke_files)
        for j in range(len(stroke_files)):
        	os.remove(stroke_files[j]) 
