import xml.etree.ElementTree as ET
from collections import Counter
import os, sys

###########################
# Check if xmls are valid #
###########################

classes = []
classNameCountArray = []

directory = '/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/images/fine_aug/'

# Read files in dir #
files = os.listdir(directory)

error = ""
# check.txt log file with not valid xml # 
with open("/home/petropoulakis/Desktop/check.txt", "w") as f:
    for filename_short in files:
        if (not filename_short.endswith(".xml") ):
            continue
        else:

            # Parse current file #
            filename = directory + "/" + filename_short
            tree = ET.parse(filename)
            size = tree.find('size')

            # Increase counter #
            classNameCountArray.append(name)

            # Read basic info #
            imageWidth = int(size.find('width').text)
            imageHeight = int(size.find('height').text)
            imageArea = imageWidth * imageHeight

            # Wrong extension #
            if (".JPG" in tree.find('filename').text):
                print filename_short,"Error .jpg to JPG"
                error = "ERROR2"

            # Add class #
            name = tree.find('object').find('name').text
            if not (name in classes):
                classes.append(name)


            # Get box #
            boundingBox = tree.find('object').find('bndbox')
            xmin = int( boundingBox.find('xmin').text )
            ymin = int( boundingBox.find('ymin').text )
            xmax = int( boundingBox.find('xmax').text )
            ymax = int( boundingBox.find('ymax').text )

            boxWidth  = xmax - xmin
            boxHeight = ymax - ymin
            boxArea = boxWidth * boxHeight

            '''
            if (boxArea < 0.01 * imageArea):
                print filename, "Too Small object"
                error = "ERROR"
            '''

            # Check box #
            if (xmin > xmax or ymin > ymax):
                print filename_short,"Invalid Min Max relationship",xmin,xmax,ymin,ymax
                error = "ERROR3"
                f.write(filename_short + "\n")

            if (xmax > imageWidth or ymax > imageHeight):
                print filename_short,"Invalid Limits of Bounding Box",xmin,xmax,ymin,ymax
                error = "ERROR4"
                f.write(filename_short + "\n")

            if (xmin < 0 or xmax < 0 or ymin < 0 or ymax < 0):
                print filename_short,"Bounding box is zero",xmin,xmax,ymin,ymax
                error = "ERROR5"
                f.write(filename_short + "\n")

# Print stats #
print "Found  " + str( len(classes) ) + " Classes = ",classes
c = Counter(classNameCountArray)
print c

if (not error):
    print "SUCCESS!"

sys.exit(error)

# Petropoulakis Panagiotis            #  
# Inspired by github tensorflow issue #
