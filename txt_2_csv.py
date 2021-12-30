import xml.etree.ElementTree
import os
import csv
import numpy as np
paths='/home/sounak/Desktop/BTP/BIT_Database_New/english_modified/'
for i in range(134):
    path=paths+str((i+1))+'/'
    text_files = [f for f in os.listdir(path) if (f.endswith('.txt'))]
    for i in range(len(text_files)):
        text_files[i]=path+text_files[i]
    for files in text_files:
        fname=files.split('.')[0]+".csv"
        with open(fname,'a') as csvfile:
            data=np.genfromtxt(files)
            writer=csv.writer(csvfile)
            writer.writerows(data)
    	csvfile.close()
'''for files in inkml_files:
    root = xml.etree.ElementTree.parse(files).getroot()
    fname=files.split('.')[0]+".csv"
    with open(fname,'a') as csvfile:
        a=np.asarray(('x','y','p','t'))
        writer = csv.writer(csvfile)
        writer.writerow(a)
        strokes = sorted(root.findall('trace'), key=lambda child: child.attrib['id'])
        for stroke in strokes:
            stroke = stroke.text.strip().split(',')
            stroke = [point.strip().split(' ') for point in stroke]
            #stroke = [[float(x), float(y), float(p), int(t)] for x, y, p, t in stroke]
            for x,y,p,t in stroke:
                a=np.asarray((x,y,p,t))
                writer = csv.writer(csvfile)
                writer.writerow(a)
    csvfile.close()'''
        


