from skimage import img_as_float
from skimage import measure
from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt
import os
import numpy as np
import imagehash
from PIL import Image
imagesName1 = []
values = [] # Keep similar images 
allNames = []

threshold = 3 # Similarity 

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
delFiles = []
values = []

# Compare images #
end = len(imagesName1)
count = 0
for i in range(end):
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

indexes = np.argsort(values)
with open(logFile, 'w') as fp:
    for i in range(len(indexes)):
    #for i in range(len(delFiles)):
    # Save result #
        fp.write("val: " + str(values[indexes[i]]) + " " + delFiles[indexes[i]][0] + " " + delFiles[indexes[i]][1] + "\n")
        #fp.write(delFiles[i] +  "\n")

