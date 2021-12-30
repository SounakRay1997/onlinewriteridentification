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
        fname=files.split('.')[0]+"_processed.csv"
        print(fname)
        with open(fname,'a') as csvfile:
            data=pd.read_csv(files)
            data=np.asarray(data)
            for k in range(data.shape[0]):
                if(data[k][1]==0):
                    continue
                else:
                    a=np.asarray((data[k][0], data[k][1], data[k][4], data[k][5], data[k][6]))
                    writer=csv.writer(csvfile)
                    writer.writerow(a)
        csvfile.close()
            
            
        
