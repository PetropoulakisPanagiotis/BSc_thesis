from skimage import img_as_float
from skimage import measure
from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt
import os
import numpy as np

imagesName1 = []
values = [] # Keep similar images 
allNames = []

threshold = 0.7 # Similarity 

# Set paths #
path1 = '/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/images/full/'
logFile = '/home/petropoulakis/Desktop/similarLog.txt'

# Read images #
files = os.listdir(path1)

for file in files:
    if(file.endswith(".jpg")):
        imagesName1.append(file)

# Compare images #
for i in range(len(imagesName1)):
    tmp1 = img_as_float(io.imread(path1 + imagesName1[i]))
    tmp1 = resize(tmp1, (tmp1.shape[0] / 10, tmp1.shape[1] / 10))

    for j in range(len(imagesName1)):
        tmp2 = img_as_float(io.imread(path1 + imagesName1[j + i + 1]))
        tmp2 = resize(tmp2, (tmp2.shape[0] / 10, tmp2.shape[1] / 10))
        ssimVal = measure.compare_ssim(tmp1, tmp2, data_range=tmp1.max() - tmp1.min(), multichannel=True)

        # Similar images #
        if(ssimVal > threshold):
            values.append(ssimVal)
            allNames.append((imagesName1[i], imagesName1[j + i + 1]))

    print i

indexes = np.argsort(values)
with open(logFile, 'w') as fp:
    for i in range(len(indexes)):
    # Save result #
        fp.write("val: " + str(values[indexes[i]]) + "names: " + allNames[indexes[i]][0] + " " + allNames[indexes[i]][1] + "\n")
