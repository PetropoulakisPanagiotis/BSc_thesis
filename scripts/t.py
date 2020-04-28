import imageio
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import os
import xml.etree.ElementTree as ET
import numpy as np

imagesDir = "/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/images/full/"
saveDir = "/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/images/new"
counter = 1409

# Read files #
files = os.listdir(imagesDir)

imagesName = []

for file in files:

    if(file.endswith(".xml")):
        continue

    imagesName.append(file)

# Augment every image #
for image in imagesName:
    currImage = imageio.imread(imagesDir + image)

    # Read xml and bounding box #
    xmlName = imagesDir + image[:-4] + ".xml"
    tree = ET.parse(xmlName)
    x1 = int(tree.find('object').find('bndbox').find('xmin').text)
    y1 = int(tree.find('object').find('bndbox').find('ymin').text)
    x2 = int(tree.find('object').find('bndbox').find('xmax').text)
    y2 = int(tree.find('object').find('bndbox').find('ymax').text)

    box = BoundingBoxesOnImage([BoundingBox(x1=x1,y1=y1,x2=x2,y2=y2)], shape=currImage.shape)
    ia.imshow(box.draw_on_image(currImage))
    # Perform augmentations #
    translate = np.arange(-0.2, 0.2, 0.012)

    for x in translate:
        if(x != 0):
            #0.85 - 0.92 #
            seq = iaa.Sequential([iaa.Pad(px=49)])
            newImage, newBox = seq(image=currImage, bounding_boxes=box)

            ia.imshow(newBox.draw_on_image(newImage))
            exit()
            # Get new boxes
            newX1 = newBox.bounding_boxes[0].x1
            newY1 = newBox.bounding_boxes[0].y1
            newX2 = newBox.bounding_boxes[0].x2
            newY2 = newBox.bounding_boxes[0].y2

            newXml = tree
            newPath = saveDir + str(counter) + "_robot"
            newXml.find('path').text = newPath + ".jpg"

            newName = str(counter) + "_robot.jpg"
            newXml.find('filename').text = newName 
            newXml.find('object').find('bndbox').find('xmin').text = str(int(newX1))
            newXml.find('object').find('bndbox').find('ymin').text = str(int(newY1))
            newXml.find('object').find('bndbox').find('xmax').text = str(int(newX2))
            newXml.find('object').find('bndbox').find('ymax').text = str(int(newY2))

            newXml.write(newPath + ".xml")
            imageio.imwrite(saveDir + newName, newImage) 
            counter += 1
