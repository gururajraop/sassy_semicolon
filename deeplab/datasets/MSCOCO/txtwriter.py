# Writes txt files containing the filenames for the training and validation 
# splits, to 

import os

f=open("train.txt",'w+')

for path, dirs, files in os.walk("./panoptic_train2017"):
    for filename in files:
        f.write(os.path.splitext(filename)[0]+ "\n")

f.close()

f=open("val.txt",'w+')

for path, dirs, files in os.walk("./panoptic_val2017"):
    for filename in files:
        if filename.startswith('.') == 0:
            f.write(os.path.splitext(filename)[0]+ "\n")

f.close()
