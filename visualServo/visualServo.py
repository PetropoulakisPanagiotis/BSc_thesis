#!/usr/bin/env python
from __future__ import division
from __future__ import print_function
import rospy
from message_filters import Subscriber, TimeSynchronizer
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import tensorflow as tf
import numpy as np
import cv2
import os
import time
import signal
import traceback
from threading import Lock
# Custom #
from objectDetector.objectDetector import objectDetector
from controller.controller import Controller
import helpers

# Disable warnings/warm up messages for tf #
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Read and handle frames #
class visualServo:

    def __init__(self):

        rospy.init_node('visual_servo', disable_signals=True)

        ##########
        # Topics #
        ##########
        self.topicColor = "/kinect2/qhd/image_color_rect"
        self.topicDepth = "/kinect2/qhd/image_depth_rect"
        self.topicVel = "/summit_xl/robotnik_base_control/cmd_vel" # Publish velocities

        #############
        # Main vars #
        #############
        self.pioneerDetector = None # MobileNet
        self.controller = None
	self.l = Lock()

        # Measure accuracy #
        self.totalFrames = 0
        self.totalLostFramesDetection = 0
        self.totalLostFramesKinect = 0 # Not available depth 

        # Color - Depth- camera #
        self.width = None # Kinect qhd: 960  
        self.height = None # Kinect qhd: 540 
        self.color = None
        self.depth = None
        # Hard coded for perforamce, else read topic #
        self.K = np.array([[540.68603516, 0, 479.75], [0, 540.68603516, 269.75], [0, 0, 1]])
        self.D = np.zeros((5, 1))

        # Chached calculations to convert a pixel to 3d coordinates # 
        self.mapY = []
        self.mapX = []

        # General #
        self.stop = True # Robot is initial stopped
        self.stableFps = True # Check if loop has at leat minHzController during experiment(when the robot start) 
        self.warmUp = False # Servo init - create map etc 
        self.debug = True
        self.excecutionCode = "Visual Servo: success"
        self.newMessages = False
        self.fps = 0
        self.minRateKinect = 1.0
        self.minHzController = 8

        # Track images #
        self.prevImage = np.zeros((500, 500, 3), dtype="uint8")
        self.currImage = np.zeros((500, 500, 3), dtype="uint8")

        # Init detector #
        self.pioneerDetector = objectDetector()

        # Init controller                   #
        # See controller module for tunning #
        self.controller = Controller()

        # Listen color and depth # 
        self.subImageColor = Subscriber(self.topicColor, Image, queue_size=5)
        self.subImageDepth = Subscriber(self.topicDepth, Image, queue_size=5)

        # Sync messages #
        self.messages = TimeSynchronizer([self.subImageColor, self.subImageDepth], queue_size=5)

        # Set callback function #
        self.messages.registerCallback(self.callbackKinect)

        # Leader velocities topic #
        self.velPub = rospy.Publisher(self.topicVel, Twist, queue_size=1)

        # Set sig handler for proper termination #
        signal.signal(signal.SIGINT, self.sigHandler)
        signal.signal(signal.SIGTERM, self.sigHandler)
        signal.signal(signal.SIGTSTP, self.sigHandler)

        if(self.debug):
            print("Visual servo start")
            print("==============================")
            print("Min controller hz: " + str(self.minHzController) + " hz")
            print("Min kinect rate: " + str(self.minRateKinect) + " sec")
            print("==============================")

    # Read messages #
    def callbackKinect(self, imageColor, imageDepth):

        # Frames #
	self.l.acquire()
        self.color = helpers.readImage(imageColor)
        self.depth = helpers.readImage(imageDepth)
        self.newMessages = True
	self.l.release()

    # Basic method: detect object, find pos, calculate and publish velocities #
    def servo(self):

        try:
            framesServo = 0 # Measure fps

            # Measure fps #
            startTimeFps = time.time()
	    count = 0

            # Measure time for new frames #
            startTimeKinect = time.time()

            ##################
            # Process frames #
            ##################
            while True and not rospy.is_shutdown():

                # Fix fps #
                nowTimeFps = time.time()
                elapsed = nowTimeFps - startTimeFps
                if(elapsed >= 1):
		    count += 1
                    startTimeFps = nowTimeFps
                    self.fps = framesServo / elapsed
                    framesServo = 0
                    #if self.debug:
                    #    print("Fps: " + str(self.fps))

                    if count > 2 and not self.stop and self.fps < self.minHzController:
                        self.stopFollower()
                        self.stop = True
                        self.stableFps = False

                ######################
                # Read current frame #
                ######################
                currColor, currDepth = self.readFrame()
                if currColor.shape == (1, 1, 3):
                    # Check rate of kinect #
                    nowTimeFps = time.time()
                    elapsed = nowTimeFps - startTimeKinect
                    if self.warmUp and elapsed > self.minRateKinect:
                        self.excecutionCode = "Visual Servo: Bad rate of kinect"
                        break
                    else:
                        continue
                else:
                    startTimeKinect = time.time() # Reset timer for kinect - new frame arrived 

                # Increase frames #
                if not self.stop:
                    self.totalFrames += 1

                # Fix frame for detector #
                colorRGB = cv2.cvtColor(currColor, cv2.COLOR_BGR2RGB)
                colorExpand = np.expand_dims(colorRGB, axis=0)

                # Display current frame #
                cv2.imshow("detect", currColor)
                cv2.waitKey(0)

                #################
                # Detect object #
                #################
                result = self.pioneerDetector.predict(colorExpand)
                if sum(score >= 0.5 for score in result["detection_scores"]) > 1 or result["detection_scores"][0] < 0.5:
                    helpers.saveImage(currColor, "/media/csl-mini-pc/TOSHIBA EXT/test7/%d.jpg" % self.totalFrames)
                    if not self.stop:
                    	self.totalLostFramesDetection += 1
                    continue

                # Draw box #
                self.pioneerDetector.visualize(currColor, result)
                self.currImage = currColor.copy()
                helpers.saveImage(currColor, "/media/csl-mini-pc/TOSHIBA EXT/test7/%d.jpg" % self.totalFrames)
                #cv2.imshow("detect", currColor)
                #cv2.waitKey(0)

                # Extract box #
                box = result["detection_boxes"][0]

                # Get real box coordinates - type int - pixels # 
                xMin, xMax, yMin, yMax, code = objectDetector.getBox(self.width, self.height, box)
                if code != "Detector: success":
                    self.excecutionCode = code
                    break

                # Check box #
                valid, code = self.pioneerDetector.validBox(xMin, xMax, yMin, yMax, self.width, self.height)
                if not valid:
                    self.excecutionCode = code
                    break

                ######################
                # Find pos of leader #
                ######################
                xPixel, yPixel = self.pioneerDetector.getCenter(xMin, xMax, yMin, yMax)

                # Display center #
                #currColor = helpers.drawCircle(currColor, xPixel, yPixel)
                #cv2.imshow("detect", currColor)
                #cv2.waitKey(0)

                xC, yC, zC = helpers.pixelToCoordinates(currDepth, xPixel, yPixel, self.mapX, self.mapY)
		if xC == 0: # Not available depth
                    if not self.stop:
                    	self.totalLostFramesKinect += 1
                    continue
                elif xC == -1:
                    self.exitCode = "Visual Servo: pixelToCoordinates bad arguments"
                    break

		# Leader pos with respect to robot frame #
                xL, yL, zL = helpers.cameraToRobot(xC, yC, zC)
		##################################
                # Compute velocities of follower #
                ##################################
                uF, omegaF, code = self.controller.calculateVelocities(xL, yL)
                if code != "Controller: success":
                    self.excecutionCode = code
                    break

                if self.stop == True and (uF != 0 or omegaF != 0):
                    self.totalFrames += 1
                    self.stop = False

                if uF == 0 and omegaF == 0:
                    self.stop = True

                # Push commands to follower topic #
                self.publishVelocities(uF, omegaF)

                # Successful loop - increase counter for fps #
                framesServo += 1

                self.prevImage = currColor.copy()

            # End while - safe exit #
            self.terminateServo()

        except Exception, e: # Exception occurred 
            print("Visual Servo: Exception occurred: ")
            print(traceback.format_exc())

            # Safe exit - stop robot #
            self.terminateServo()

    # Read current frame #
    def readFrame(self):
	self.l.acquire()
        if(self.newMessages == True):
            self.newMessages = False
            print("sdsd\n")
            currColor = self.color.copy()
            if currColor.size == 0:
		self.l.release()
                return np.zeros([1,1,3], dtype=np.uint8), np.zeros([1,1,3], dtype=np.uint8)

            currDepth = self.depth.copy()
            if currDepth.size == 0:
		self.l.release()
                return np.zeros([1,1,3], dtype=np.uint8), np.zeros([1,1,3], dtype=np.uint8)

	    self.l.release()
            # Fix map # 
            if(not self.warmUp):

                # Fix vars #
                self.width = currColor.shape[1]
                self.height = currColor.shape[0]

                # Create map for fast calculations # 
                self.mapX, self.mapY = helpers.createMap(self.K, self.width, self.height)

                if(len(self.mapX) == 0):
                    return np.zeros([1,1,3], dtype=np.uint8), np.zeros([1,1,3], dtype=np.uint8)

                self.warmUp = True
        else:
	    self.l.release()
            return np.zeros([1,1,3], dtype=np.uint8), np.zeros([1,1,3], dtype=np.uint8)

        return currColor, currDepth

    # Publish velocities to robot topic #
    def publishVelocities(self, u, omega):
        velMsg = Twist()

        velMsg.linear.x = u
        velMsg.linear.y = 0.0
        velMsg.linear.z = 0.0

        velMsg.angular.x = 0.0
        velMsg.angular.y = 0.0
        velMsg.angular.z = omega

        # Publish velocities #
        self.velPub.publish(velMsg)

    # Stop robot function for safety #
    def stopFollower(self):
        velMsg = Twist()

        velMsg.linear.x = 0.0
        velMsg.linear.y = 0.0
        velMsg.linear.z = 0.0

        velMsg.angular.x = 0.0
        velMsg.angular.y = 0.0
        velMsg.angular.z = 0.0

        count = 0
        while count < 10:
            self.velPub.publish(velMsg)
            count += 1

    def terminateServo(self):
        # Save last two images #
        helpers.saveImage(self.currImage, "./currImage.jpg")
        helpers.saveImage(self.prevImage, "./prevImage.jpg")

        # Stop robot #
        self.stopFollower()
        self.stop = True

        # Stop ros #
        rospy.signal_shutdown(0)

        self.printStats()
        exit()

    # Handle signals for proper termination #
    def sigHandler(self, num, frame):

        print("Visual Servo: Signal occured:  " + str(num))

        # Safe exit - stop robot #
        self.terminateServo()

    # Result of visual servo #
    def printStats(self):
        print("Excecution Code: " + self.excecutionCode)
        print("Warm up: " + str(self.warmUp))
        print("Stable fps: " + str(self.stableFps))
        print("Robot stop: " + str(self.stop))
        print("Total frames: " + str(self.totalFrames))
        print("Total lost frames due to detection: " + str(self.totalLostFramesDetection))
        print("Total lost frames due to kinect: " + str(self.totalLostFramesKinect))

if __name__ == "__main__":

    # Init #
    robotServo = visualServo()

    # Detect target #
    robotServo.servo()

# Petropoulakis Panagiotis
