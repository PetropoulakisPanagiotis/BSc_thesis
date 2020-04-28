from skimage import img_as_float
from skimage import measure
from skimage import io
from skimage.transform import resize
from matplotlib import pyplot as plt
import os
import numpy as np
from imagehash import ImageHash, hex_to_hash
import imagehash
from PIL import Image

if False:
    path1 = '/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/images/full/'
    logFile = '/home/petropoulakis/Desktop/checksum.txt'

    # Read images #
    files = os.listdir(path1)
    hashes = []
    names = []
    for file in files:
        if(file.endswith(".jpg")):
            hashes.append(imagehash.dhash(Image.open(path1 + file), hash_size=32))
            names.append(file)


    #indexes = np.argsort(values)
    with open(logFile, 'w') as fp:
    #    for i in range(len(indexes)):
        for i in range(len(hashes)):
            fp.write(str(hashes[i]) + " " + names[i] + "\n")

if True:
    hashes = []
    names = []
    with open("/home/petropoulakis/Desktop/checksum.txt", 'r') as f:
        line = f.readline()
        while line:
            x = line.split()
            hashes.append(hex_to_hash(x[0]))
            names.append(x[1])
            line = f.readline()

    path1 = '/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/images/val/'
    hashCmp = imagehash.dhash(Image.open(path1 + "170_robot.jpg"), hash_size=32)

    val = []
    for i in range(len(hashes)):
        val.append(hashCmp - hashes[i])

    indexes = np.argsort(val)
    with open("/home/petropoulakis/Desktop/result_170.txt", 'w') as fp:
        for i in range(len(indexes)):
            fp.write(str(val[indexes[i]]) + " " + names[indexes[i]]  + "\n")
