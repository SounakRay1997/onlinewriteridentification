import os
import numpy as np
import pandas as pd
import csv
path='/home/sounak/Desktop/BTP/Dataset/BIT_Database_New/english_modified/Test/'
for i in range(134):
    stroke_counter=0
    paths=path+str((i+1))+'/'
    csv_files=[f for f in os.listdir(paths) if (f.endswith('.csv'))]
    for j in range(len(csv_files)):
        csv_files[j]=paths+csv_files[j]
    for files in csv_files:
        folder_name=files.split('.')[0]+'/'
        os.mkdir(folder_name)
        stroke_counter=0
        data=pd.read_csv(files)
        data=np.asarray(data)
        count_of_consecutive_ones=0
        for i in range(data.shape[0]):
            #fname=files.split('.')[0]+'_strokes_'+str(stroke_counter)+'.csv'
            #print(fname)
            #with open(fname,'a') as csvfile:
            if (data[i][1]%2==0):
                count_of_consecutive_ones=0
                continue
            else:
                if (count_of_consecutive_ones==0):
                    count_of_consecutive_ones=count_of_consecutive_ones+1
                    stroke_counter=stroke_counter+1
                    fname=folder_name+str(stroke_counter)+'_strokes'+'.csv'
                    print(fname)
                    with open(fname,'a') as csvfile:
                        a=np.asarray(('time', 'pen_up_down', 'x', 'y', 'pressure'))
                        writer=csv.writer(csvfile)
                        writer.writerow(a)
                        a=np.asarray((data[i][0], data[i][1], data[i][4], data[i][5], data[i][6]))
                        writer=csv.writer(csvfile)
                        writer.writerow(a)
                    csvfile.close()
                else:
                    with open(fname,'a') as csvfile:
                        a=np.asarray((data[i][0], data[i][1], data[i][4], data[i][5], data[i][6]))
                        writer=csv.writer(csvfile)
                        writer.writerow(a)
                    csvfile.close()
                
                
           
     
