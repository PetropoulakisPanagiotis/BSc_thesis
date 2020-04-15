#!/usr/bin/env python
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

# Read and handle frames #
class kinectHandler:

    def __init__(self):

        # Paths for detections #
        #self.modelPath = "/home/csl-mini-pc/Desktop/petropoulakis/TensorFlow/workspace/robot_detection/model/frozen_inference_graph.pb"
        #self.labelPath = "/home/csl-mini-pc/Desktop/petropoulakis/TensorFlow/workspace/robot_detection/annotations/label_map.pbtxt"

        self.modelPath = "/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/model/frozen_inference_graph.pb"
        self.labelPath = "/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/annotations/label_map.pbtxt"

        # Set topic names #
        self.topicColor = "/kinect2/qhd/image_color_rect"
        self.topicDepth = "/kinect2/qhd/image_depth_rect"
        self.topicCameraInfoColor = "/kinect2/qhd/camera_info"
        self.topicCameraInfoDepth = "/kinect2/qhd/camera_info"

        self.color = None
        self.depth = None
        self.cameraMatrixColor = None
        self.cameraMatrixDepth = None

        self.totalFrames = 0
        self.lostFrames = 0
        self.falsePositives = 0
        self.errorCode = 0

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

        # Initialize detector #
        self.pioneerDetector = objectDetector(self.modelPath, self.labelPath)

        # Set sig handler #
        signal.signal(signal.SIGINT, self.sigHandler)

    def callback(self, imageColor, imageDepth, cameraInfoColor, cameraInfoDepth):

        # Read messages and get the content #
        self.messagesMutex.acquire()
        helpers.readCameraInfo(cameraInfoColor, self.cameraMatrixColor)
        helpers.readCameraInfo(cameraInfoDepth, self.cameraMatrixDepth)
        self.color = helpers.readImage(imageColor)
        self.depth = helpers.readImage(imageDepth)
        self.newMessages = True
        self.messagesMutex.release()

    # Detect given object and publish (x, y) coordinates of box #
    def detect(self):
        savePath = "/home/petropoulakis/Desktop/experiment/boxes/"
        frameCount = 0
        fpsText = ""
        type = 0

        #color = (255, 255, 255)
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #fontScale = 1
        #thickness = 1
        #org = (5, 30)

        # Start listening messages #
        self.threadListener = threading.Thread(target=helpers.threadListenerFunc)
        self.threadListener.start()

        start = time.time()
        # Process frames #
        while(1):

            # Read current frame #
            self.messagesMutex.acquire()
            if(self.newMessages == True):
                currColor = self.color
                currDepth = self.depth
                self.newMessages = False
            else:
                self.messagesMutex.release()
                continue

            self.messagesMutex.release()

            frameCount += 1
            now = time.time()
            elapsed = now - start
            if(elapsed >= 1):
                fps = frameCount / elapsed
                frameCount = 0
                start = now
                print(fps)
		print("\n")

            # Fix frame #
            colorFixed = cv2.cvtColor(currColor, cv2.COLOR_BGR2RGB)
            colorExpand = np.expand_dims(colorFixed, axis=0)

            # Detect #
            result = self.pioneerDetector.predict(colorExpand)
            self.totalFrames += 1

         #   if sum(score >= 0.5 for score in result["detection_scores"]) > 1 or result["detection_scores"][0] > 0.5:

            self.pioneerDetector.visualize(colorFixed, result)
            colorFixed = cv2.cvtColor(colorFixed, cv2.COLOR_BGR2RGB)
            helpers.saveImage(colorFixed, savePath + str(self.totalFrames) + ".jpg")

        rospy.signal_shutdown("Closing kinect handler\n")

        # Wait thread #
        self.threadListener.join()

    def sigHandler(self, num, frame):
        print("Total frames: " + str(self.totalFrames) +"\n")
        print("Lost frames: " + str(self.lostFrames) + "\n")
        print("False positives frames: " + str(self.falsePositives))

        rospy.signal_shutdown("Closing kinect handler\n")

        # Wait thread #
        self.threadListener.join()
        exit()

if __name__ == "__main__":
    rospy.init_node('kinect_handler')

    kinect =  kinectHandler()

    kinect.detect()

# Petropoulakis Panagiotis
