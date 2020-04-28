import xml.etree.ElementTree as ET
import functools
import os, sys
import shutil

def myCompare(x, y):
    x = x.partition("_")[0] # Extract the number 
    y = y.partition("_")[0]
    x = int(x)
    y = int(y)

    return (x - y)

directory = '/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/images/train/'
newDir = '/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/images/train/'

files = os.listdir(directory)

files = os.listdir(directory)
filesAll = []

for file in files:
    if(file.endswith(".xml")):
        filesAll.append(file)

count = 1
filesAll.sort(key=functools.cmp_to_key(myCompare))

for file in filesAll:

    filename = directory + file
    tree = ET.parse(filename)
    tree.find('filename').text = str(count) + "_robot.jpg"
    tree.find('path').text = newDir + str(count) + "_robot.jpg"
    f = '/home/petropoulakis/Desktop/plain/' + str(count) + "_robot.xml"
    count += 1
    tree.write(f)
