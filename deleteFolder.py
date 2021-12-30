import os
import shutil
path='/home/sounak/Desktop/BTP/Dataset/BIT_Database_New/chinese_english_modified/Test/'
for i in range(133):
    paths=path+str((i+1))+'/'
    csv_files=[f for f in os.listdir(paths) if (f.endswith('.csv'))]
    for j in range(len(csv_files)):
        csv_files[j]=paths+csv_files[j]
    for files in csv_files:
        folder_name=files.split('.')[0]+'/'+'PLGF_9_window_size/'
        shutil.rmtree(folder_name)
