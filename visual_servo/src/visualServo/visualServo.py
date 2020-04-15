#!/usr/bin/env python
from __future__ import division
from __future__ import print_function
import rospy
import signal
from message_filters import Subscriber, TimeSynchronizer
from sensor_msgs.msg import Image, CameraInfo
from threading import Lock
import threading
import thread
import cv2
from cv_bridge import CvBridge
import helpers
import time
from objectDetector.objectDetector import objectDetector
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image as imagePil
from cv2 import aruco 
import matplotlib  as mpl
import os
import tensorflow as tf
import math 
from geometry_msgs.msg import Twist
import open3d as o3d

# Disable warnings tf #
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Change backend for plt #
plt.switch_backend('WXAgg')

# Read and handle frames #
class visualServo:

    def __init__(self):

        # Paths #
        self.modelPath = "/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/model/frozen_inference_graph.pb"
        self.labelPath = "/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/annotations/label_map.pbtxt"
        self.templatePath = "/home/petropoulakis/catkin_ws/src/visual_servo/data/1.pcd"
        self.templatePcd = o3d.io.read_point_cloud(self.templatePath) 
        
        #o3d.visualization.draw_geometries([self.templatePcd])

        # Topics #
        self.topicColor = "/kinect2/qhd/image_color_rect"
        self.topicDepth = "/kinect2/qhd/image_depth_rect"
        self.topicCameraInfoColor = "/kinect2/qhd/camera_info"
        self.topicCameraInfoDepth = "/kinect2/qhd/camera_info"
        self.topicVelocity = ""

        # Main vars #
        self.width = 960
        self.height = 540
        self.maxDepth = 1.2 # Max deviation of depth 
        self.minAreaBox = 20 
        self.maxOffsetBox = 22 # Box must be inside the image by offset 
        self.offsetBox = 20  # Get a bigger box than the original. Must be less than max offset  
        self.mapY = [] # Chached calculations to convert pixel to 3d coordinate - x component
        self.mapX = [] # Y - component 
        self.templatePcd = None # Template pcd for object to find orientation by registration
        self.color = None # Current frames 
        self.depth = None
        self.cameraMatrixColor = None
        self.cameraDistortColor = None
        self.K = None
        self.D = None
        
        # Servo parameters #
        self.kapa = 0.01 
        self.gama = 0.005 
        self.eStar = 1.7 # Desired error 
        self.maxError = 5.0
        self.minError = 1.0
        self.boxOffset = 5 # Take a bigger RoI of the detected box by (boxOffset) pixel   

        # General #
        self.debug = True
        self.printFps = False
        self.success = "True"
        self.totalFrames = 0
        self.badDetections = 0
        self.totalLostFrames = 0 
        self.lostFrames = 0 # Per 100 frames
        self.maxLostFrames = 20 # Max lost rate for 100 frames 
        self.lostConsecutiveFrames = 0
        self.maxLostConsecutiveFrames = 6  
        self.depthDev = 0.2 # Meters
        self.prevDepth = 0.0
        self.badPointsTotal = 0
        self.badDevTotal = 0
        
        # Lock for messages #
        self.messagesMutex = Lock()
        self.newMessages = False

        # Listen color and depth # 
        self.subImageColor = Subscriber(self.topicColor, Image, queue_size=1)
        self.subImageDepth = Subscriber(self.topicDepth, Image, queue_size=1)
        self.subCameraInfoColor = Subscriber(self.topicCameraInfoColor, CameraInfo, queue_size=1)
        self.subCameraInfoDepth = Subscriber(self.topicCameraInfoDepth, CameraInfo, queue_size=1)

        # Sync messages #
        self.messages = TimeSynchronizer([self.subImageColor, self.subImageDepth, self.subCameraInfoColor, self.subCameraInfoDepth], queue_size=1)

        # Set callback function #
        self.messages.registerCallback(self.callback)

        # Velocity #
        #self.velPublisher = rospy.Publisher(self.topicVelocity, Twist, queue_size=1)

        # Initialize detector #
        self.pioneerDetector = objectDetector(self.modelPath, self.labelPath)

        # Start listening messages #
        self.threadListener = threading.Thread(target=helpers.threadListenerFunc)
        self.threadListener.start()

        # Set sig handler #
        signal.signal(signal.SIGINT, self.sigHandler)
        signal.signal(signal.SIGTSTP, self.sigHandler)

    # Read camera messages #
    def callback(self, imageColor, imageDepth, cameraInfoColor, cameraInfoDepth):

        self.messagesMutex.acquire()
        
        # Camera Matrix #
        self.cameraMatrixColor, self.cameraDistortColor = helpers.readCameraInfo(cameraInfoColor)
        self.cameraMatrixDepth, self.cameraDistortDepth = helpers.readCameraInfo(cameraInfoDepth)
        
        # Frames #
        self.color = helpers.readImage(imageColor)
        self.depth = helpers.readImage(imageDepth)
        
        self.newMessages = True
        self.messagesMutex.release()

    # Detect object and calculate velocities #
    def servo(self):
        frameRate = 0 # Measure rate of our control
        countReadCameraMatrix = 0
        depthDevFlag = 0

        # Set marker #
        #markers = helpers.getMarker()
        
        # Measure fps #
        startTimeFps = time.time()

        # Measure time for new frames #
        startTimeFrame = time.time()

        ##################
        # Process frames #
        ##################
        while True and not rospy.is_shutdown():

            # Check state #
            if (self.lostFrames > self.maxLostFrames or self.lostConsecutiveFrames > self.maxLostConsecutiveFrames):
                
                if self.lostFrames > self.maxLostFrames:
                    self.success = "lost frames"
                else:
                    self.success = "lost consecutive frames"
                break

            # Check if we can't receive frames #
            nowTimeFps = time.time()
            elapsed = nowTimeFps - startTimeFrame
            if(elapsed >= 2.0):
                self.success = "Can't receive frames"
                break
 
            # Read current frame #
            self.messagesMutex.acquire()
            if(self.newMessages == True):
                currColor = self.color.copy()
                currDepth = self.depth.copy()
                self.newMessages = False
           
                # Read color camera matrix and fix map # 
                if(countReadCameraMatrix == 0):
                    self.K = self.cameraMatrixColor
                    self.D = np.zeros((5,1))
                    self.K = np.array([[self.K[0], self.K[1], self.K[2]], [self.K[3], self.K[4], self.K[5]], [self.K[6], self.K[7], self.K[8]]])
                    
                    # Create map for fast calculations # 
                    self.mapX, self.mapY = helpers.createMap(self.K, self.width, self.height) 
                    countReadCameraMatrix = 1
            else:
                self.messagesMutex.release()
                continue

            self.messagesMutex.release()

            # Empty frame #
            if currColor.size == 0:
                continue

            # Bad K #
            if np.count_nonzero(self.K) == 0:
                continue

            startTimeFrame = time.time() # Reset timer for new frames 

            self.totalFrames += 1

            # Reset frames #                   
            if self.totalFrames != 100 and self.totalFrames % 100 == 0:   
                self.lostFrames = 0
            
            # Fix fps #
            nowTimeFps = time.time()
            elapsed = nowTimeFps - startTimeFps
            if(elapsed >= 1):
                fps = frameRate / elapsed
                startTimeFps = nowTimeFps
                frameRate = 0
                if self.printFps:
                    print("Fps: " + str(fps))

            # Fix frame for detector #
            colorRGB = cv2.cvtColor(currColor, cv2.COLOR_BGR2RGB)
            colorExpand = np.expand_dims(colorRGB, axis=0)
          
            #################
            # Detect object #
            #################
            result = self.pioneerDetector.predict(colorExpand)
            if sum(score >= 0.5 for score in result["detection_scores"]) > 1 or result["detection_scores"][0] < 0.5:
                continue
           
            self.pioneerDetector.visualize(currColor, result)
                
            cv2.imshow("detect", currColor)
            cv2.waitKey(0)
         
            # Extract box #
            box = result["detection_boxes"][0]
            # Get real box coordinates - type int #  
            xMin, xMax, yMin, yMax = objectDetector.getBox(self.width, self.height, box)
            if helpers.validBox(self.maxOffsetBox, self.minAreaBox, xMin, xMax, yMin, yMax, self.width, self.height):
                continue 
         
            # Create new bigger box #
            xMin, xMax, yMin, yMax = helpers.getNewBox(self.boxOffset, xMin, xMax, yMin, yMax)

            # Create point cloud for box #
            pcd = helpers.createPCD(colorRGB, currDepth, self.mapX, self.mapY, xMin, xMax, yMin, yMax, self.maxDepth)   
            if pcd == None:
                continue

            o3d.visualization.draw_geometries([pcd])
            o3d.io.write_point_cloud("../../data/37.pcd", pcd)

            #x, y, z, theta = helpers.getPose()

            '''
            if ids == None or len(ids) == 0: 
                self.badDetections += 1
                self.lostConsecutiveFrames += 1
                self.totalLostFrames += 1
                self.lostFrames += 1
                continue
 
            
            # Check center # 
            if helpers.checkOutOfBounds(centerX, centerY, self.width, self.height):
                self.success = "Detection out of bounds" 
                break

            # Find 3D coordinates for center with respect to the camera #
            success, cameraX, cameraY, cameraZ = helpers.getXYZ(self.K, centerX, centerY, currDepth, self.width, self.height)
            if success == False:
                self.badPointsTotal += 1
                self.lostFrames += 1
                self.lostConsecutiveFrames += 1
                self.totalLostFrames += 1
                continue

            # Fix initial depth #
            if depthDevFlag == 0:
                self.prevDepth = cameraZ
                depthDevFlag = 1

            # Check deviation of depth # 
            if cameraZ > self.prevDepth + self.depthDev or cameraZ < self.prevDepth - self.depthDev:
                self.badDevTotal += 1
                self.lostFrames += 1
                self.lostConsecutiveFrames += 1
                self.totalLostFrames += 1
                self.prevDepth = cameraZ
                continue

            #################################################
            # Perform servoing. Publish velocities to robot #
            ################################################
            success = self.servo(cameraX, cameraY, cameraZ)
            if success == False:
                self.success = "Servo out of bounds" 
                break
           
            # Fix stats #
            frameRate += 1

            if self.lostConsecutiveFrames != 0:
                self.lostConsecutiveFrames = 0
       
        '''
            break

        # Stop robot #
        self.publishVelocities(0.0, 0.0)

        # Terminating #
        rospy.signal_shutdown("Closing kinect handler\n")

        # Wait thread #
        self.threadListener.join()
    
        self.printStats()
    
    # Find and publish robot velocities with position based visual servo #
    def controller(self, cameraX, cameraY, cameraZ):

        robotX, robotY, robotZ = helpers.convertCameraToRobot(cameraX, cameraY, cameraZ)
        
        # Current error #
        e = math.sqrt(robotY ** 2 + robotX ** 2) 

        # Check if we lost the target #
        if e > self.maxError or e < self.minError:
            return False
        
        # Angle from the goal #
        a = math.asin(robotY / e)
        
        # Controller # 
        robotU = self.gama * math.cos(a) * (e - self.eStar) 
        robotOmega = self.kapa * a          

        if self.debug:
            print('X: {:.4} Y: {:.4} Z: {:.4} e: {:.4} a: {:.4} u: {:.4} omega: {:.4}'.format(robotX, robotY, robotZ, e, math.degrees(a), robotU, robotOmega)) 

        # Publish velocities #
        self.publishVelocities(robotU, robotOmega)

        return True

    # Find position of object by point cloud registration(with template)
    def estimatePos():
        pass

    # Estimate velocity of object #
    def estimateVelocity():
        pass

    # Publish velocities to robot topic #
    def publishVelocities(self, u, omega):
        velMsg = Twist()

        # Normalize velocities if needed #
        u, omega = helpers.normalizeVelocity(u, omega)

        velMsg.linear.x = u
        velMsg.linear.y = 0.0
        velMsg.linear.z = 0.0

        velMsg.angular.x = omega
        velMsg.angular.y = 0.0

        #self.velPublisher.publish(velMsg)
  
    # Handle signals for proper termination #
    def sigHandler(self, num, frame):

        # Stop robot #
        self.publishVelocities(0.0, 0.0)
 
        rospy.signal_shutdown("Closing kinect handler\n")
 
        # Wait thread #
        self.threadListener.join()

        self.printStats()
        exit()

    def printStats(self):
        print("Success: " + self.success)
        print("Total frames: " + str(self.totalFrames))
        print("Total lost frames: " + str(self.totalLostFrames))
        print("Total lost frames due to detection: " + str(self.badDetections))
        print("Total lost frames due to bad points: " + str(self.badPointsTotal))
        print("Total lost frames due to bad deviation: " + str(self.badDevTotal))

if __name__ == "__main__":
    rospy.init_node('visual_servo')

    # Init #
    robotServo = visualServo()
    
    # Perform visual servoing #
    robotServo.servo()

# Petropoulakis Panagiotis
