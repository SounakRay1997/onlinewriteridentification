import csv
import os
import numpy as np
import pandas as pd
path='/home/sounak/Desktop/BTP/Dataset/BIT_Database_New/chinese_english_modified/Test/'
for i in range(133):
    paths=path+str((i+1))+'/'
    csv_files=[f for f in os.listdir(paths) if (f.endswith('.csv'))]
    for j in range(len(csv_files)):
        csv_files[j]=paths+csv_files[j]
    for files in csv_files:
        line_number=1
        folder_name=files.split('.')[0]+'/'
        stroke_files=[f for f in os.listdir(folder_name) if (f.endswith('.csv'))] 
        for j in range(len(stroke_files)):
            stroke_files[j]=folder_name+stroke_files[j]
        print(stroke_files)
        for j in range(len(stroke_files)):
            print(stroke_files[j])
            data=pd.read_csv(stroke_files[j])
            data=np.asarray(data)
            '''x_data=data[:,2]
            y_data=data[:,3]
            len_data_x=x_data.shape[0]
            len_data_y=y_data.shape[0]
            processed_list=[]'''
            fname=stroke_files[j].split('.')[0]+"_processed.csv"
            print(fname)
            with open(fname,'a') as csvfile:
                writer=csv.writer(csvfile)
                b=np.asarray(('time', 'pen_up_down', 'x', 'y', 'pressure'))
                writer.writerow(b)
                count=0.0
                sump=0.0
                for k in range(data.shape[0]-1):
                    #sump=0
                    if(data[k,2]==data[k+1,2] and data[k,3]==data[k+1,3]):
                        sump+=data[k,4]
                        count=count+1
                    else:
                        sump+=data[k,4]
                        count+=1
                        sump/=count     
                        a=np.asarray((data[k,0],data[k,1],data[k,2],data[k,3],sump))
                        count=0
                        sump=0
                        writer.writerow(a)
                k=(data.shape[0]-1)
                if count==0:
                    a=np.asarray((data[k,0],data[k,1],data[k,2],data[k,3],data[k,4]))
                    writer.writerow(a)
                else:
                    a=np.asarray((data[k,0],data[k,1],data[k,2],data[k,3],(sump+data[k,4])/(count+1)))
                    writer.writerow(a)
            csvfile.close()
