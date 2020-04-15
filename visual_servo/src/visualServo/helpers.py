from __future__ import division
import cv2
from cv2 import aruco 
import rospy
import math
import numpy as np
import open3d as o3d
import copy
from cv_bridge import CvBridge

# Listen messages #
def threadListenerFunc():
    rospy.spin()

# Read camera matrix #
def readCameraInfo(cameraInfo):
    return cameraInfo.K, cameraInfo.D

# Read image from message #
def readImage(imageMessage):

    # Bridge with ros and opencv #
    bridge = CvBridge()

    # Read and convert image to opencv type #
    return bridge.imgmsg_to_cv2(imageMessage, imageMessage.encoding)

# Create fast mapping for pixel to 3d coordinate #
def createMap(K, width, height):
    mapX = []
    mapY = []
    
    fx = K[0][0]
    fy = K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    for x in range(width):
        mapX.append((x - cx) * (1.0/fx))        

    for y in range(height):
        mapY.append((y - cy) * (1.0/fy))          

    return mapX, mapY

# Assume that the box is out of bounds #
def validBox(maxOffsetBox, minAreaBox, xMin, xMax, yMin, yMax, width, height):
   
    if (xMax - xMin) * (yMax - yMin) < minAreaBox:
        return True

    if (xMax - xMin) < 20 or (yMax - yMin) < 20:
        return True

    if xMin < maxOffsetBox or xMax > width - maxOffsetBox or yMin < maxOffsetBox or yMax > height - maxOffsetBox:
        return True
    
    return False

# Create bigger box from the original #
def getNewBox(offset, xMin, xMax, yMin, yMax):
    xMin = xMin - offset
    xMax = xMax + offset
    yMin = yMin - offset
    yMax = yMax + offset
    
    return xMin, xMax, yMin, yMax

# Find depth of a pixel. Use some neighbors to find a more accurate depth #
# X, Y must not be quite close to the limits of the image                 #
def estimateDepthPixel(depth, x, y):
    Z = 0
    pointsZ = [] 

    # Scan neighbors #
    for i in range(y - 1, y + 1):
        for j in range(x - 2, x + 2):

            currDepth = depth[y][x] / 1000.0

            if currDepth != 0:
                pointsZ.append(currDepth)

    if len(pointsZ) < 5:
        return Z

    Z = sum(pointsZ) / len(pointsZ)

    return Z

# Convert point to robot frame #
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

# Fix max velocities #
def normalizeVelocity(u, omega):

    if np.abs(u) > 0.02:
        if u > 0:
            u = 0.02
        else:
            u = -0.02

    if np.abs(omega) > 0.02:
        if omega > 0:
            omega = 0.02
        else:
            omega = -0.02

    return u, omega

# Rotation matrix to euler angles(only yaw) #
def rotationToEuler(R):
    
    if isClose(R[2,0], -1.0):
        yaw = math.pi/2.0

    elif isClose(R[2,0], 1.0):
        yaw = -math.pi/2.0
 
    else:
        yaw = -math.asin(R[2,0])

    return yaw

def isClose(x, y, rtol=1.e-5, atol=1.e-8):

    return abs(x-y) <= atol + rtol * abs(y)

# Save opencv type image #
def saveImage(image, path):
    params = []
    params.append(cv2.IMWRITE_JPEG_QUALITY)
    params.append(100)
    params.append(cv2.IMWRITE_PNG_COMPRESSION)
    params.append(0)
    params.append(cv2.IMWRITE_PNG_STRATEGY)
    params.append(cv2.IMWRITE_PNG_STRATEGY_RLE)

    cv2.imwrite(path, image, params)

# Petropoulakis Panagiotis
