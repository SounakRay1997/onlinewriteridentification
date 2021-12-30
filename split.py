import os
import shutil
import random
paths='/home/sounak/Desktop/BTP/BIT_Database_New/english_modified/'
train_path='/home/sounak/Desktop/BTP/Dataset/BIT_Database_New/english_modified/Train/'
test_path='/home/sounak/Desktop/BTP/Dataset/BIT_Database_New/english_modified/Test/'
for i in range(134):
    path=paths+str((i+1))+'/'
    list_of_files=[0, 1, 2]
    csv_files = [f for f in os.listdir(path) if (f.endswith('.csv'))]
    a=random.randint(0, 2)
    #b=random.randint(3, 5)
    csv_files_whole_path=[]
    for j in range(len(csv_files)):
        csv_files_whole_path.append(path+csv_files[j])
    test_file_path=test_path+str(i+1)+'/'+csv_files[a]
    shutil.copy(csv_files_whole_path[a], test_file_path)
    #test_file_path=test_path+str(i+1)+'/'+csv_files[b]
    #shutil.copy(csv_files_whole_path[b], test_file_path)
    list_of_files.remove(a)
    #list_of_files.remove(b)
    for k in range(2):
        train_file_path=train_path+str(i+1)+'/'+csv_files[list_of_files[k]]
        shutil.copy(csv_files_whole_path[list_of_files[k]], train_file_path)
        
     
        
    
    
