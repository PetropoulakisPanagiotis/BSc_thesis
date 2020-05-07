from __future__ import division
import rospy
import cv2
import math
import numpy as np
from cv_bridge import CvBridge

# Read camera matrix #
def readCameraInfo(cameraInfo):
    return cameraInfo.K, cameraInfo.D

# Read image from message #
def readImage(imageMessage):

    # Bridge with ros and opencv #
    bridge = CvBridge()

    # Read and convert image to opencv type #
    return bridge.imgmsg_to_cv2(imageMessage, imageMessage.encoding)

# Create fast mapping for pixel to 3d coordinates #
def createMap(K, width, height):
    mapX = []
    mapY = []

    if(width <= 0 or height <= 0 or K.shape != (3,3)):
        return [], []   
 
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    for x in range(width):
        mapX.append((x - cx) * (1.0/fx))        

    for y in range(height):
        mapY.append((y - cy) * (1.0/fy))          

    return mapX, mapY

# Check if box is out of bounds #
def validBox(minOffsetBox, minAreaBox, xMin, xMax, yMin, yMax, width, height):

    if(minOffsetBox <= 0 or minAreaBox <= 0 or width <= 0 or height <= 0 or xMin < 0 or xMax >= width or yMin < 0 or yMax >= height):
        return False, -1

    if (xMax - xMin) * (yMax - yMin) < minAreaBox:
        return False, -2

    if xMin < minOffsetBox or xMax > width - minOffsetBox or yMin < minOffsetBox or yMax > height - minOffsetBox:
        return False, -2
    
    return True, 0

# Create bigger box from the original #
# Use valid box if needed             #
# Input(in pixels)                    #
def getNewBox(offset, xMin, xMax, yMin, yMax):
    xMin = xMin - offset
    xMax = xMax + offset
    yMin = yMin - offset
    yMax = yMax + offset
    
    return xMin, xMax, yMin, yMax

# Find depth of a pixel. Use some neighbors for better accuracy #
# X, Y must not be quite close to the limits of the image       #
def estimateDepthPixel(depth, x, y):
    Z = 0
    pointsZ = [] 

    if(depth.size == 0 or x - 2 < 0 or x + 2 >= depth.shape[1] or y - 1 < 0 or y + 1 >= depth.shape[0]):
        return 0.0 

    # Scan neighbors #
    for i in range(y - 1, y + 1):
        for j in range(x - 2, x + 2):

            currDepth = depth[y][x] / 1000.0

            if currDepth != 0:
                pointsZ.append(currDepth)

    if len(pointsZ) < 5:
        return 0.0

    Z = sum(pointsZ) / len(pointsZ)

    return Z

# Convert point to robot frame             #
# Depends on the relative pos camera-robot #
def convertCameraToRobot(cX, cY, cZ):
    translation = np.asarray([[-0.2953], [-0.1154], [1.0215]])
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0, -1.0, 0.0]])
    rz = np.asarray([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0, 0, 1.0]])

    c = np.asarray([[cX], [cY], [cZ]])
    r = np.dot(np.dot(rz, rx), c) + translation 

    rX = r[0][0]
    rY = r[1][0]
    rZ = r[2][0]

    return rX, rY, rZ

# Fix velocities(m/sec) #
def normalizeVelocity(u, omega, maxU=0.02, maxOmega=0.02):

    if(maxU <= 0 or maxOmega <= 0):
        return None, None

    if np.abs(u) > maxU:
        if u > 0:
            u = maxU
        else:
            u = -maxU

    if np.abs(omega) > maxOmega:
        if omega > 0:
            omega = maxOmega
        else:
            omega = -maxOmega

    return u, omega

# Rotation matrix to euler angles(only yaw) #
def rotationToEuler(R):
   
    if(R.shape != (3,3) or R[2,0] > 2 * math.pi or R[2,0] < -2 * math.pi):
        return None
     
    if isClose(R[2,0], -1.0):
        yaw = math.pi/2.0
    elif isClose(R[2,0], 1.0):
        yaw = -math.pi/2.0 
    else:
        yaw = -math.asin(R[2,0])

    return yaw

# For rotationToEuler #
def isClose(x, y, rtol=1.e-5, atol=1.e-8):
    return abs(x-y) <= atol + rtol * abs(y)

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
