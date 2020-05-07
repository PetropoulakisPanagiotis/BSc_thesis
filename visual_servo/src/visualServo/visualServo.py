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
import pclProcessing 
import helpers

# Disable warnings/warm up messages for tf #
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Change backend for plt #
plt.switch_backend('WXAgg')

# Read and handle frames #
class visualServo:

    def __init__(self):

        rospy.init_node('visual_servo', disable_signals=True)
        
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
        self.maxDepth = 0.8 # Cut point away from the detected box(meters)
        self.minAreaBox = 40 # Smaller box is false detection  
        self.minOffsetBox = 25 # Box must be inside the image by this offset(pixels) 
        self.boxOffset = 10  # Generate a bigger box from the detected to secure that leader lies inside   
        
        # Set specifications for experiment #
        self.servoDistance = 1.7 # Goal distance from leader - controller
        self.maxX = 5.0 # Target lost (meters) 
        self.minX = 1.0 
        self.devY = 2.5 
        self.devZ = 2.0 
        self.devTheta = math.pi/2 
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
        self.deltaT = 500.0
 
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
        self.messages.registerCallback(self.callbackKinect)

        # Velocity #
        #self.velPublisher = rospy.Publisher(self.topicVelocity, Twist, queue_size=1)

        # Angle #
        #self.subImageColor = Subscriber(self.topicColor, Image, queue_size=1)
        
        # Init detector #
        self.pioneerDetector = objectDetector(self.modelPath, self.labelPath)
        
        # Init template #
        self.templatePcd = o3d.io.read_point_cloud(self.templatePath) 

        # Start listening messages #
        self.threadListener = threading.Thread(target=self.listenerFunc)
        self.threadListener.start()

        if(self.debug):
            self.__str__()   

    # Listener thread function #
    def listenerFunc(self):
        rospy.spin()
 
    # Read messages #
    def callbackKinect(self, imageColor, imageDepth, cameraInfoColor, cameraInfoDepth):

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

            #if self.debug:
            #    print("---------------------------------------")

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
            if((elapsed >= 500.0 and self.xPrev != 0) or (self.xPrev != 0 and elapsed  > self.deltaT)):  
                self.excecutionCode = "Bad rate of loop"
                break

            ######################
            # Read current frame #
            ######################
            currColor, currDepth = self.readFrame()
            if(currColor.shape == (1,1,3)):
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

            valid, code = helpers.validBox(self.minOffsetBox, self.minAreaBox, xMin, xMax, yMin, yMax, self.width, self.height)
            if code == -1:
                self.excecutionCode = "Invalid arguments: validBox" 
                break      

            if not valid:
                self.excecutionCode = "Lost target" 
                break
            
            # Find position of leader with point cloud registration #
            xL, yL, zL, thetaL, transformationNew, typeError, code = self.estimatePos(colorRGB, currDepth, xMin, xMax, yMin, yMax)

            if typeError == 1:
                if code == -1:
                    self.excecutionCode = "Invalid arguments: estimation pos - createPCD" 
                    break
                else:
                    self.totalInvalidPoints += 1    
                    continue 

            if typeError == 2:
                if code == -1:
                    self.excecutionCode = "Invalid arguments: estimate pos - estimationPos" 
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
            
            # First loop - initialize previous pos # 
            if(self.xPrev == 0):
                # Update pos # 
                self.xPrev = xL
                self.yPrev = yL
                self.thetaPrev = thetaL

                # Loop was succesful - reset vars #
                self.estimationPrev = transformationNew
                self.totalLostFrames -= 1
                self.lostFrames -= 1
                self.lostConsecutiveFrames = 0  
                continue

            # Estimate velocity of leader #
            uL, omegaL, code = self.estimateVelocity(self.xPrev, xL, self.yPrev, yL, self.thetaPrev, thetaL)
            if code == -1:
                self.excecutionCode = "Invalid arguments: estimateVelocity" 
                break

            if code == -2:
                self.excecutionCode = "Invalid arguments: big difference in posPrev and posNew" 
                break

            # Update prev pos # 
            self.xPrev = xL
            self.yPrev = yL
            self.thetaPrev = thetaL
            self.estimationPrev = transformationNew

            # Calculate and publish velocity command - follower #
            thetaF = 0
            sucessVel = self.controller(xL, yL, zL, thetaL, thetaF, uL, omegaL)
            if not sucessVel:
                self.excecutionCode = "Invalid arguments: controller" 
                break
          
            # Loop was succesful - reset vars #
            self.totalLostFrames -= 1
            self.lostFrames -= 1
            self.lostConsecutiveFrames = 0             

        # End while - Success - Leader stopped #
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
        uF = (uL * math.cos(thetaL - thetaF)) + \
             (k1 * ((math.cos(thetaF) * xError) + (math.sin(thetaF) * yError)))
      
        # Fix limits #
        if abs(thetaL - thetaF) <= 0.05:
            tmp = 1
        else:
            tmp = (math.sin(thetaL - thetaF) / (thetaL - thetaF)) 
 
        omegaF = omegaL + \
                 (k2 * uL * tmp * (math.cos(thetaF) * xError - math.sin(thetaF) * yError)) + \
                 (k3 * (thetaL - thetaF))

        if self.debug:
            print('Follower velocity(uF: {:.4} m/s omegaF: {:.4} m/s)'.format(uF, omegaF)) 

        # Publish velocities #
        #self.publishVelocities(uF, omegaF)

        return True

    # Min/Max values depends on this specific experiment #
    # Input: follower frame                              #
    def lostTarget(self, x, y, z, theta, maxX, minX, devY, devZ, devTheta):
        return False, 0
        if(maxX <= 0 or minX <= 0 or devY <= 0 or devZ <= 0 or devTheta < -2 * math.pi or devTheta > 2 * math.pi):          
            return True, -1

        if(x > maxX or x < minX or abs(y) > devY or abs(z) > devZ or abs(theta) > devTheta):
            return True, 0

        else:
            return False, 0

    # Find pos of object by point cloud registration(with template) #
    # Result in follower frame                                      #
    # Min/Max values depends on this specific experiment            #
    def estimatePos(self, colorRGB, currDepth, xMin, xMax, yMin, yMax):
        
        # Create point cloud for box #
        pcd, code = pclProcessing.createPCD(colorRGB, currDepth, self.mapX, self.mapY, xMin, xMax, yMin, yMax, self.maxDepth)   
        if pcd == None:
            return -1, -1, -1, -1, None, 1, code # 1-> createPCD fails + code 
 
        cX, cY, cZ, theta, transformationNew, code = pclProcessing.estimatePos(self.templatePcd, pcd, self.estimationPrev) 
        if code != 0:
            return -1, -1, -1, -1, None, 2, code # 2-> estimatePos fails + code 
 
        x, y, z = helpers.convertCameraToRobot(cX, cY, cZ)

        # Target is far away #
        lost, code = self.lostTarget(x, y, z, theta, self.maxX, self.minX, self.devY, self.devZ, self.devTheta)
        if lost: 
            return -1, -1, -1, -1, None, 3, code

        if self.debug:
            print('Leader pos: (xL: {:.4} m yL: {:.4} m zL: {:.4} m thetaL: {:.4} degrees)'.format(x, y, z, math.degrees(theta)))             

        return x, y, z, theta, transformationNew, 0, 0

    # Estimate velocity of object #
    def estimateVelocity(self, x, xNew, y, yNew, theta, thetaNew, devX=0.15, devY=0.15, devTheta=0.34):

        if(self.deltaT <= 0):
            return None, None, -1 

        if(abs(x - xNew) > devX or abs(y - yNew) > devY or abs(theta - thetaNew) > devTheta):
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
        
        if self.debug:
            print('Leader velocity: (uL: {:.4} m/s omegaL: {:.4} m/s)'.format(uL, omegaL))             

        return uL, omegaL, 0
 
    # Read current frame #
    def readFrame(self):
    
        self.messagesMutex.acquire()
        if(self.newMessages == True):
            
            currColor = self.color.copy()
            if currColor.size == 0:
                return np.zeros([1,1,3], dtype=np.uint8), np.zeros([1,1,3], dtype=np.uint8)
           
            currDepth = self.depth.copy()
            self.newMessages = False
       
            # Read color camera matrix and fix map # 
            if(self.xPrev == 0):
                
                self.K = self.cameraMatrixColor
                if np.count_nonzero(self.K) == 0:
                    self.K = None
                    return np.zeros([1,1,3], dtype=np.uint8), np.zeros([1,1,3], dtype=np.uint8)
                
                self.width = currColor.shape[1]
                self.height = currColor.shape[0]
                self.D = np.zeros((5,1))
                self.K = np.array([[self.K[0], self.K[1], self.K[2]], [self.K[3], self.K[4], self.K[5]], [self.K[6], self.K[7], self.K[8]]])
                
                # Create map for fast calculations # 
                self.mapX, self.mapY = helpers.createMap(self.K, self.width, self.height)
                if(len(self.mapX) == 0):
                    self.K = None
                    return np.zeros([1,1,3], dtype=np.uint8), np.zeros([1,1,3], dtype=np.uint8)
        
        else:
            self.messagesMutex.release()
            return np.zeros([1,1,3], dtype=np.uint8), np.zeros([1,1,3], dtype=np.uint8)

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
        if(self.xPrev != 0):
            self.publishVelocities(0.0, 0.0)

        # Terminating #
        rospy.signal_shutdown("Terminating\n")
       
        # Wait thread #
        self.threadListener.join()
 
        # Stop ros #
        rospy.signal_shutdown(0)
    
        self.printStats()

    # Handle signals for proper termination #
    def sigHandler(self, num, frame):

        print("Signal occured:  " + str(num))

        # Stop robot #
        self.publishVelocities(0.0, 0.0)
 
        rospy.signal_shutdown("Terminating\n")
 
        # Wait thread #
        self.threadListener.join()

        # Stop ros #
        rospy.signal_shutdown(0)

        self.printStats()

        exit()

    def printStats(self):
        print("Excecution Code: " + self.excecutionCode)
        print("Total frames: " + str(self.totalFrames))
        print("Total lost frames: " + str(self.totalLostFrames))
        print("Total lost frames due to detection: " + str(self.totalBadDetections))
        print("Total lost frames due to registration: " + str(self.totalBadRegistrations))
        print("Total lost frames due to invalid number of points in point cloud: " + str(self.totalInvalidPoints))
  
    # Info about visual servo #      
    def __str__(self):
        print("Visual servo parameters:")
        print("Goal distance from follower: " + str(self.servoDistance))
        print("Max permitted x: " + str(self.maxX))
        print("Min permitted x: " + str(self.minX))
        print("Permitted deviation of y: " + str(self.devY))
        print("Permitted deviation of z: " + str(self.devZ))
        print("Permitted deviation of theta: " + str(self.devTheta))
        print("Velocity deltaT: " + str(self.deltaT))
        print('Gains: (z: {:.4}  b: {:.4})'.format(self.z, self.b))      
        print("Max permitted lost frames in 100 frames: " + str(self.maxLostFrames))
        print("Max permitted lost consecutive frames: " + str(self.maxLostConsecutiveFrames))
        print("Max depth deviation allowed in detected box: " + str(self.maxDepth))
        print("Min area of box: " + str(self.minAreaBox))
        print("Min offset of box from bounds: " + str(self.minOffsetBox))
        print("Box offset from the detected box: " + str(self.boxOffset))
   
if __name__ == "__main__":

    # Init #
    robotServo = visualServo()

    # Perform visual servoing  #
    # Detect target and follow #
    robotServo.servo()

# Petropoulakis Panagiotis
