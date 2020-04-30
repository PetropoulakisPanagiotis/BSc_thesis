#!/usr/bin/env python
from __future__ import division
from __future__ import print_function
import rospy
from message_filters import Subscriber, TimeSynchronizer
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import os
import thread
import threading
from threading import Lock
import time
import signal
import math 
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib  as mpl
import tensorflow as tf
import open3d as o3d
# Custom #
from objectDetector.objectDetector import objectDetector
from pclProcessing import *
import helpers

# Disable warnings/warm up messages for tf #
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Change backend for plt #
plt.switch_backend('WXAgg')

# Read and handle frames #
class visualServo:

    def __init__(self):

        #########
        # Paths #
        #########
        self.modelPath = "/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/model/frozen_inference_graph.pb"
        self.labelPath = "/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/annotations/label_map.pbtxt"
        self.templatePath = "/home/petropoulakis/catkin_ws/src/visual_servo/data/template.pcd"
        
        ##########
        # Topics #
        ##########
        self.topicColor = "/kinect2/qhd/image_color_rect"
        self.topicDepth = "/kinect2/qhd/image_depth_rect"
        self.topicCameraInfoColor = "/kinect2/qhd/camera_info"
        self.topicCameraInfoDepth = "/kinect2/qhd/camera_info"
        self.topicOrientation = "" # Follower robot 
        self.topicVelocity = ""    # Follower robot 

        #############
        # Main vars #
        #############
        self.pioneerDetector = None
        self.templatePcd = None # 3D model of leader - for pos estimation 
        self.maxDepth = 1.2 # Cut point away from the detected box(meters)
        self.minAreaBox = 40 # Smaller box is false detection  
        self.maxOffsetBox = 25 # Box must be inside the image by this offset(pixels) 
        self.boxOffset = 10  # Generate a bigger box from the detected to secure that leader lies inside   
        
        # Set specifications for experiment #
        self.servoDistance = 1.7 # Goal distance from leader - controller
        maxX = 5.0 # Target lost (meters) 
        minX = 1.0 
        devY = 2.5 
        devZ = 2.0 
        devTheta = math.pi/2 
        self.totalFrames = 0
        self.totalLostFrames = 0 
        self.totalBadDetections = 0
        self.totalInvalidPoints = 0 # Not enough/surplus points to build point cloud of object
        self.totalBadRegistrations = 0 # Registration failed
        self.lostFrames = 0 # Per 100 frames
        self.lostConsecutiveFrames = 0
        self.maxLostFrames = 20 # Max lost rate for 100 frames 
        self.maxLostConsecutiveFrames = 6  
        
        # Color - Depth- camera #
        self.width = None # Kinect qhd: 960  
        self.height = None # Kinect qhd: 540 
        self.color = None 
        self.depth = None
        self.cameraMatrixColor = None
        self.cameraDistortColor = None
        self.K = None # Camera matrix 
        self.D = None

        # Chached calculations to convert a pixel to 3d coordinate # 
        self.mapY = [] 
        self.mapX = []
        
        # Servo parameters #
        self.z = 0.1 # Range: (0, 1) 
        self.b = 0.2 # Range: > 0
        
        # Worst excecution time per loop(sec)      #
        # Necessary for velocity estimation        #
        # Depends on current hadware,              #
        # accuracy of detection and pos estimation #
        self.deltaT = 0.1
 
        # Track pos #  
        self.xPrev = 0 
        self.yPrev = 0
        self.thetaPrev = 0        
        self.estimationPrev = np.identity(4) # For registration
        
        # General #
        self.debug = True
        self.printFps = False
        self.save = False
        self.excecutionCode = "Success"

        # Lock for messages #
        self.messagesMutex = Lock()
        self.newMessages = False

        # Set sig handler for proper termination #
        signal.signal(signal.SIGINT, self.sigHandler)
        signal.signal(signal.SIGTERM, self.sigHandler)
        signal.signal(signal.SIGTSTP, self.sigHandler)

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

        # Angle #
        #self.subImageColor = Subscriber(self.topicColor, Image, queue_size=1)
        
        # Init detector #
        self.pioneerDetector = objectDetector(self.modelPath, self.labelPath)
        
        # Init template #
        self.templatePcd = o3d.io.read_point_cloud(self.templatePath) 

        # Start listening messages #
        self.threadListener = threading.Thread(target=helpers.threadListenerFunc)
        self.threadListener.start()

    # Read messages #
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

    # Basic method: detect object, find pos, calculate and publish velocities #
    def servo(self):
        frameRate = 0 # Measure rate of our control

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
                    self.excecutionCode = "Max lost frames exceeded"
                else:
                    self.excecutionCode = "Max lost consecutive frames exceeded"
                break

            # Check rate of loop                         #
            # Can't read topics or after warm up(1 loop) #  
            # excecution is too slow                     # 
            nowTimeFps = time.time()
            elapsed = nowTimeFps - startTimeFrame
            if((elapsed >= 2.0) or (self.xPrev != 0 and elapsed  > self.deltaT)):  
                self.excecutionCode = "Bad rate of loop"
                break

            ######################
            # Read current frame #
            ######################
            currColor, currDepth = self.readFrame()
            if(currColor == None)
                continue
            
            startTimeFrame = time.time() # Reset timer for new frames 

            # Fix vars #
            self.totalFrames += 1
            self.totalLostFrames += 1 # Assume this loop will fail 
            self.lostFrames += 1
            self.lostConsecutiveFrames += 1

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
                self.totalBadDetections += 1
                continue 
 
            #self.pioneerDetector.visualize(currColor, result)
            #cv2.imshow("detect", currColor)
            #cv2.waitKey(0)
         
            # Extract box #
            box = result["detection_boxes"][0]

            # Get real box coordinates - type int #  
            xMin, xMax, yMin, yMax = objectDetector.getBox(self.width, self.height, box)
         
            # Create new bigger box #
            xMin, xMax, yMin, yMax = helpers.getNewBox(self.boxOffset, xMin, xMax, yMin, yMax)
            valid, code = helpers.validBox(self.maxOffsetBox, self.minAreaBox, xMin, xMax, yMin, yMax, self.width, self.height)
            if code == -1:
                self.excecutionCode = "Invalid arguments: validBox" 
                break      

            if not valid:
                self.excecutionCode = "Lost target" 
                break

            # Find position of leader with point cloud registration #
            xNew, yNew, zNew, thetaNew, transformationNew, typeError, code = self.estimatePos(colorRGB, currDepth, xMin, xMax, yMin, self.estimationPrev):
            
            if typeError == 1:
                if code == -1:
                    self.excecutionCode = "Invalid arguments: estimation pos" 
                    break
                else:
                    self.totalInvalidPoints += 1    
                    continue 

            if typeError == 2:
                if code == -1:
                    self.excecutionCode = "Invalid arguments: estimatePos" 
                    break
                else:
                    self.totalBadRegistrations += 1    
                    continue 

            if typeError == 3:
                if code == -1:
                    self.excecutionCode = "Invalid arguments: lostTarget" 
                    break
                else:
                    self.excecutionCode = "Lost target" 
                    break

            # Estimate velocity of leader #
            uL, omegaL, code = self.estimateVelocity(self.xPrev, xNew, self.yPrev, yNew, self.thetaPrev, thetaNew):

            if code == -1:
                self.excecutionCode = "Invalid arguments: estimateVelocity" 
                break

            if code == -2:
                self.excecutionCode = "Invalid arguments: big difference in x,y,z and xNew,yNew,thetaNew" 
                break

            # Calculate and publish velocity command - follower #
            sucessVel = self.controller(self, xL, yL, zL, thetaL, thetaF, uL, omegaL)
            if not sucessVel:
                self.excecutionCode = "Invalid arguments: controller" 
                break
          
            # Update pos # 
            self.xPrev = xNew
            self.yPrev = yNew
            self.thetaPrev = thetaNew
            self.estimationPrev = transformationNew

            # Loop was succesful - reset vars #
            self.totalLostFrames -= 1
            self.lostFrames -= 1
            self.lostConsecutiveFrames = 0             

        # End while - Success #
        self.terminateServo()
 
    # Calculate and publish robot velocities with position based visual servo #
    # Input must be with respect to the follower frame                        #
    # Check input with lostTarget function                                    #
    # Controller: Samson C 1993 Time-varying feedback stabilization of        #
    # car-like wheeled mobile robots                                          #
    def controller(self, xL, yL, zL, thetaL, thetaF, uL, omegaL):

        if self.servoDistance <= 0:
            return False

        if(self.z <= 0 or self.z >= 1 or self.b <= 0):
            return False   
 
        # Position initialization            #
        # Follow robot with a fixed distance #
        xError = self.servoDistance - xL
        yError = 0 - yL 

        # Fix gains #
        k1 = k3 = 2 * self.z * math.sqrt((omegaL ** 2) + (self.b * (uL ** 2)))
        k2 = self.b

        # Calculate velocities of follower #
        uF = (uL * math.cos(thetaL - thetaF)) +
             (k1 * ((math.cos(thetaF) * xError) + (math.sin(thetaF) * yError)))
      
        # Fix limits #
        if abs(thetaL - thetaF) <= 0.05:
            tmp = 1
        else:
            tmp = (math.sin(thetaL - thetaF) / (thetaL - thetaF)) 
 
        omegaF = omegaL +
                 (k2 * uL * tmp * (math.cos(thetaF) * xError - math.sin(thetaF) * yError)) +
                 (k3 * (thetaL - thetaF))

        if self.debug:
            print('u: {:.4} omega: {:.4}'.format(uF, omegaF)) 

        # Publish velocities #
        #self.publishVelocities(uF, omegaF)

        return True

    # Min/Max values depends on this specific experiment #
    # Input: follower frame                              #
    def lostTarget(self, x, y, z, theta, maxX, minX, devY, devZ, devTheta):

        if(maxX <= 0 or minX <= 0 or devY <= 0 or devZ <= 0 or devTheta < -2 * math.pi or devTheta > 2 * math.pi):          
            return True, -1

        if(x > maxX or x < minX or abs(y) > devY or abs(z) > devZ or abs(theta) > devTheta):
            return True, 0

        else
            return False, 0

    # Find pos of object by point cloud registration(with template) #
    # Result in follower frame                                      #
    # Min/Max values depends on this specific experiment            #
    def estimatePos(self, colorRGB, currDepth, xMin, xMax, yMin, yMax):
        
        # Create point cloud for box #
        pcd, code = pclProcessing.createPCD(colorRGB, currDepth, self.mapX, self.mapY, xMin, xMax, yMin, yMax, self.maxDepth)   
        if pcd == None:
            return -1, -1, -1, None, 1, code # 1-> createPCD fails + code 

        cX, cY, cZ, theta, t, code = pclProcessing.estimatePos(self.templatePcd, pcd, self.estimationPrev) 
        if t == None:
            return -1, -1, -1, None, 2, code # 2-> estimatePos fails + code 

        x, y, z = helpers.convertCameraToRobot(cX, cY, cZ)

        # Target is far away #
        lost, code = self.lostTarget(x, y, z, theta, self.maxX, self.minX, self.devY, self.devTheta):
        if lost: 
            return -1, -1, -1, None, 3, code

        if self.debug:
            print('x: {:.4} y: {:.4} z: {:.4} theta: {}'.format(x, y, z, math.degrees(theta)))             

        return x, y, z, theta, t, 0, 0

    # Estimate velocity of object #
    def estimateVelocity(self, x, xNew, y, yNew, theta, thetaNew, devX=0.1, devY=0.1, devTheta=0.27):

        if(self.deltaT <= 0):
            return None, None, -1 

        if abs(x - xNew) > devX or abs(y - yNew) > devY or abs(theta - thetaNew) > devTheta):
            return None, None, -2     
   
        if(math.cos(theta) == 0):
            uL = (yNew - y) / self.deltaT * math.sin(theta)
        elif(math.sin(theta) == 0):
            uL = (xNew - x) / self.deltaT * math.cos(theta)
        else:
            tmp1 = (xNew - x) / self.deltaT * math.cos(theta)
            tmp2 = (yNew - y) / self.deltaT * math.sin(theta)
            uL = (tmp1 + tmp2) / 2
        
        omegaL = (thetaNew - theta) / self.deltaT

        return uL, omegaL, 0
 
    # Read current frame #
    def readFrame(self):
    
        self.messagesMutex.acquire()
        if(self.newMessages == True):
            
            currColor = self.color.copy()
            if currColor.size == 0:
                return None, None
           
            currDepth = self.depth.copy()
            self.newMessages = False
       
            # Read color camera matrix and fix map # 
            if(self.K == None):
                
                self.K = self.cameraMatrixColor
                if np.count_nonzero(self.K) == 0:
                    self.K = None
                    return None, None
                
                self.width = currColor.shape[1]
                self.height = currColor.shape[0]
                self.D = np.zeros((5,1))
                self.K = np.array([[self.K[0], self.K[1], self.K[2]], [self.K[3], self.K[4], self.K[5]], [self.K[6], self.K[7], self.K[8]]])
                
                # Create map for fast calculations # 
                self.mapX, self.mapY = helpers.createMap(self.K, self.width, self.height)
                if(len(self.mapX) == 0):
                    self.K = None
                    return None, None 
        
        else:
            self.messagesMutex.release()
            return None, None

        self.messagesMutex.release()

        return currColor, currDepth

    # Publish velocities to robot topic #
    def publishVelocities(self, u, omega):
        velMsg = Twist()

        # Normalize velocities if needed #
        u, omega = helpers.normalizeVelocity(u, omega, maxU=0.02, maxOmega=0.02)

        velMsg.linear.x = u
        velMsg.linear.y = 0.0
        velMsg.linear.z = 0.0

        velMsg.angular.x = omega
        velMsg.angular.y = 0.0

        #self.velPublisher.publish(velMsg)

    def terminateServo(self):

        # Stop robot #
        if(self.prevX != 0):
            self.publishVelocities(0.0, 0.0)

        # Terminating #
        rospy.signal_shutdown("Terminating\n")
        
        # Wait thread #
        self.threadListener.join()
    
        self.printStats()

    # Handle signals for proper termination #
    def sigHandler(self, num, frame):

        # Stop robot #
        self.publishVelocities(0.0, 0.0)
 
        rospy.signal_shutdown("Terminating\n")
 
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
