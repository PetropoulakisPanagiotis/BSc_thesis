from __future__ import division
import rospy
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import math

# Read image from ros message #
def readImage(imageMessage):

    # Bridge with ros and opencv #
    bridge = CvBridge()

    # Read and convert image to opencv type #
    return bridge.imgmsg_to_cv2(imageMessage, imageMessage.encoding)

# Create fast mapping for pixel to 3d coordinates #
def createMap(K, width, height):
    mapX = []
    mapY = []

    if(width <= 0 or height <= 0 or K.shape != (3, 3)):
        return [], []

    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    if fx == 0 or fy == 0:
        return [], []

    for x in range(width):
        mapX.append((x - cx) * (1.0/fx))

    for y in range(height):
        mapY.append((y - cy) * (1.0/fy))

    return mapX, mapY

# Find x, y of a pixel. Use some neighbors for better accuracy #
# X, Y must not be quite close to the limits of the image      #
def pixelToCoordinates(depth, x, y, mapX, mapY, xOffset=5, yOffset=5, minPoints=3):
    X = 0
    Y = 0
    Z = 0
    pointsX = []
    pointsY = []
    pointsZ = []

    if(depth.size == 0 or x - xOffset < 0 or x + xOffset >= depth.shape[1] or y - xOffset < 0 or y + xOffset >= depth.shape[0]):
        return -1.0, 0.0, 0.0

    # Scan neighbors #
    for i in range(y - yOffset, y + yOffset):
        for j in range(x - xOffset, x + xOffset):

            currDepth = depth[y][x] / 1000.0
            if currDepth != 0:
                pointsZ.append(currDepth)
                pointsX.append(mapX[j] * currDepth)
                pointsY.append(mapY[i] * currDepth)

    if len(pointsX) < minPoints:
        return 0.0, 0.0, 0.0

    X = sum(pointsX) / len(pointsX)
    Y = sum(pointsY) / len(pointsY)
    Z = sum(pointsZ) / len(pointsZ)

    return X, Y, Z

# Debug center of box #
def drawCircle(image, x, y):
    image = cv2.circle(image, (x,y), 10, (255,0,0), 2)
    return image

# Convert point to robot frame              #
# Depends on the relative pose camera-robot #
# Camera frame: front Z, right X, down Y    #
# Robot frame: up Z, front X, left Y        #
# Angles: right handed                      # 
def cameraToRobot(cX, cY, cZ):
    translation = np.asarray([[-0.33], [-0.12], [0.64]])
    q = np.asarray([-0.49999, 0.499601, -0.499999, 0.500398])
    r = R.from_quat(q)
    r = r.as_dcm()

    c = np.asarray([[cX], [cY], [cZ]])
    result = np.dot(r, c) + translation

    X = result[0][0]
    Y = result[1][0]
    Z = result[2][0]

    return X, Y, Z

# Save opencv type image #
def saveImage(image, path):
    if(path == "" or image.size == 0):
        return False

    params = []
    params.append(cv2.IMWRITE_JPEG_QUALITY)
    params.append(100)
    params.append(cv2.IMWRITE_PNG_COMPRESSION)
    params.append(0)
    params.append(cv2.IMWRITE_PNG_STRATEGY)
    params.append(cv2.IMWRITE_PNG_STRATEGY_RLE)

    cv2.imwrite(path, image, params)

    return True

# Petropoulakis Panagiotis
