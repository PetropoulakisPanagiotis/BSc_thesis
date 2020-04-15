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
from PIL import Image as imagePil

# Read and handle frames #
class kinectHandler:

    def __init__(self):

        # Paths for detections #
        self.modelPath = "/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/model/frozen_inference_graph.pb"
        self.labelPath = "/home/petropoulakis/Desktop/TensorFlow/workspace/robot_detection/annotations/label_map.pbtxt"
        self.width = 960
        self.height = 540

        # Set topic names #
        self.topicColor = "/kinect2/qhd/image_color_rect"
        self.topicDepth = "/kinect2/qhd/image_depth_rect"
        self.topicCameraInfoColor = "/kinect2/qhd/camera_info"
        self.topicCameraInfoDepth = "/kinect2/qhd/camera_info"

        self.color = None
        self.depth = None
        self.cameraMatrixColor = None
        self.cameraMatrixDepth = None

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
        self.cameraMatrixColor = helpers.readCameraInfo(cameraInfoColor)
        
        self.cameraMatrixDepth = helpers.readCameraInfo(cameraInfoDepth)
        self.color = helpers.readImage(imageColor)
        self.depth = helpers.readImage(imageDepth)
        self.newMessages = True
        self.messagesMutex.release()

    # Detect given object and publish (x, y) coordinates of box #
    def detect(self):
        frameCount = 0
        #fpsText = ""
        #type = 0

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

            # Fix frame #
            colorFixed = cv2.cvtColor(currColor, cv2.COLOR_BGR2RGB)
            colorExpand = np.expand_dims(colorFixed, axis=0)

            type = 0
            # Detect #
            result = self.pioneerDetector.predict(colorExpand)
            
            xc, yc = self.pioneerDetector.getCenter(self.width, self.height, result["detection_boxes"][0])

            # Find angle for the center #
            K = np.array([[self.cameraMatrixColor[0], self.cameraMatrixColor[1],self.cameraMatrixColor[2]],[self.cameraMatrixColor[3], self.cameraMatrixColor[4],self.cameraMatrixColor[5]], [self.cameraMatrixColor[6], self.cameraMatrixColor[7],self.cameraMatrixColor[8]]])
            Ki = np.linalg.inv(K)
            r1 = Ki.dot([self.width / 2.0 , self.height / 2.0, 1.0])       
            
            r2 = Ki.dot([xc, yc, 1.0])       

            cosAngle = r1.dot(r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)) 
            angleRadians = np.arccos(cosAngle)

            now = time.time()
            elapsed = now - start
            if(elapsed >= 1):
                fps = frameCount / elapsed
                start = now
                frameCount = 0
                print(fps)

            if sum(score >= 0.5 for score in result["detection_scores"]) > 1:
                type = 3

            elif result["detection_scores"][0] < 0.65:
                if result["detection_scores"][0] < 0.5:
                    type = 1
                else:
                    type = 2

            print(type)
        rospy.signal_shutdown("Closing kinect handler\n")

        # Wait thread #
        self.threadListener.join()

    def sigHandler(self, num, frame):
        rospy.signal_shutdown("Closing kinect handler\n")

        # Wait thread #
        self.threadListener.join()
        exit()

if __name__ == "__main__":
    rospy.init_node('kinect_handler')

    kinect =  kinectHandler()

    kinect.detect()

# Petropoulakis Panagiotis
