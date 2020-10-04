import os
import shutil
import functools

######################
# Move and fix files #
######################
def myCompare(x, y):
    x = x.partition(".")[0] # Extract the number 
    y = y.partition(".")[0]
    x = int(x)
    y = int(y)

    return (x - y)

dir = '/home/petropoulakis/Desktop/media/15/images'
newDir = '/home/petropoulakis/Desktop/media/15/image'

count = 1

# Read files of dir #
files = os.listdir(dir)

# Keep xml files #
filesAll = []

for file in files:
    if(file.endswith(".jpg")):
        filesAll.append(file)

filesAll.sort(key=functools.cmp_to_key(myCompare))

for file in filesAll:

    shutil.move(dir + "/" + file, newDir + "/" + str(count) + ".jpg")

    count += 1
# Petropoulakis Panagiotis #
