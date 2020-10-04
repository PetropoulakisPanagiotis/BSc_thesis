import os
import functools

# Comparison for images        #
# Expected name xxxx_robot.jpg #
def myCompare(x, y):
    x = x.partition("a")[0] # Extract the number 
    y = y.partition("a")[0]
    x = int(x)
    y = int(y)

    return (x - y)

# Set paths #
imagesPath = "/home/petropoulakis/Desktop/x_val/"
newPath = "/home/petropoulakis/Desktop/x/"
baseName = "_robot.jpg"

# Initial id to be saved # 
counter = 883

# Extract images from path #
files = os.listdir(imagesPath)
images = []

for file in files:
    images.append(file)

# Sort images #
images.sort(key=functools.cmp_to_key(myCompare))

# Rename images - operation is like mv in linux #
for i in range(len(images)):
    currentName = str(counter) + baseName
    os.rename(imagesPath + images[i], newPath + currentName)
    counter += 1

# Petropoulakis Panagiotis 
