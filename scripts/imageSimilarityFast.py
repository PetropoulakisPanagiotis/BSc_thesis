import os
from PIL import Image
import numpy as np
import imagehash

imagesName1 = []
delFiles = [] # Similar files 
values = [] # Keep similar images distance  
threshold = 3

# Set paths #
path1 = '/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/images/full/'
logFile = '/home/petropoulakis/Desktop/similarLog.txt'

# Read images #
files = os.listdir(path1)
hashes = []
for file in files:
    if(file.endswith(".jpg")):
        imagesName1.append(file)
        #hashes.append(imagehash.dhash(Image.open(path1 + file), hash_size=32))
        hashes.append(imagehash.dhash(Image.open(path1 + file), hash_size=16))


# Compare images #
count = 0
for i in range(len(imagesName1)):
    if count == 5000:
        print(i)
        count = 0

    count += 1
    for j in range(end):
        if(j <= i):
            continue

        totalVal = hashes[i] - hashes[j]
        if(totalVal <= threshold):
            #delFiles.append(imagesName1[j])
            delFiles.append((imagesName1[j], imagesName1[i]))
            values.append(totalVal)

# Save similar images in log #
indexes = np.argsort(values)
with open(logFile, 'w') as fp:
    for i in range(len(indexes)):
        fp.write("val: " + str(values[indexes[i]]) + " " + delFiles[indexes[i]][0] + " " + delFiles[indexes[i]][1] + "\n")

# Petropoulakis Panagiotis #
