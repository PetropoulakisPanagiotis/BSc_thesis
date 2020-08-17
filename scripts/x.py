import os
import functools
import xml.etree.ElementTree as ET
import os, sys
import shutil

# Comparison for images        #
# Expected name xxxx_robot.jpg #
def myCompare(x, y):
    x = x.partition("_")[0] # Extract the number 
    y = y.partition("_")[0]
    x = int(x)
    y = int(y)

    return (x - y)

# Set paths #
imagesPath = "/home/petropoulakis/Desktop/train/"
newPath = "/home/petropoulakis/Desktop/train/"
baseName = "_robot.jpg"
#baseName1 = "_robot.jpg"

# Extract images from path #
files = os.listdir(imagesPath)
images = []
#images1 = []

for file in files:
  #  if(file.endswith(".xml")):
 #       images.append(file)
 #   elif file.endswith(".jpg"):
    images.append(file)

# Sort images #
#images.sort(key=functools.cmp_to_key(myCompare))
#images1.sort(key=functools.cmp_to_key(myCompare))

# Rename images - operation is like mv in linux #
for i in range(len(images)):
    currentName = baseName
    os.rename(imagesPath + images[i], newPath + currentName)
    #currentName = str(counter) + baseName1
    #os.rename(imagesPath + images1[i], newPath + currentName)