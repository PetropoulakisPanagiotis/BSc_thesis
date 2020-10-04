from __future__ import division
import rospy
from message_filters import Subscriber
from geometry_msgs.msg import Twist
import numpy as np
import time
import random
import signal
import math


# Call ros #
rospy.init_node('bot', disable_signals=True)

# Leader #
topicVel = "/robot_b/robotnik_base_control/cmd_vel/"

# Publish velocities commands #
pubVel = rospy.Publisher(topicVel, Twist, queue_size=1)

def publishVelocities(u, omega):
    velMsg = Twist()

    velMsg.linear.x = u
    velMsg.linear.y = 0.0
    velMsg.linear.z = 0.0

    velMsg.angular.x = 0.0
    velMsg.angular.y = 0.0
    velMsg.angular.z = omega

    # Publish velocities #
    pubVel.publish(velMsg)

# Stop robot #
def stopLeader():
    velMsg = Twist()

    velMsg.linear.x = 0.0
    velMsg.linear.y = 0.0
    velMsg.linear.z = 0.0

    velMsg.angular.x = 0.0
    velMsg.angular.y = 0.0
    velMsg.angular.z = 0.0

    count = 0
    while count < 30:
        pubVel.publish(velMsg)
        count += 1

# Handle signals for proper termination #
def sigHandler(num, frame):

    print("Signal occurred:  " + str(num))

    # Stop robot #
    stopLeader()

    # Close ros #
    rospy.signal_shutdown(0)
    exit()

# Set sig handler for proper termination #
signal.signal(signal.SIGINT, sigHandler)
signal.signal(signal.SIGTERM, sigHandler)
signal.signal(signal.SIGTSTP, sigHandler)

# Constant linear #
linearVel = 0.55
endTime = time.time() + 60 * 4 # Run for 4 min the experiment

# Create bucket of angular velocities #
samplesAngular = set()
num = 0.4
while num <= 2.0:

    samplesAngular.add(num)
    num += 0.5

# Pick initial random angular velocity #
angularVel = random.sample(samplesAngular, 1)[0]

count = 0
# Take random actions #
while time.time() < endTime:

    # Expondential decay #
    if abs(angularVel) > 0.05:
	angularVel *= 0.999977
    else:

        # Reset angular velocity #
        angularVel = random.sample(samplesAngular, 1)[0]

        # Change sign #
        count += 1
        if count == 1:
            angularVel *= -1
            count = 0

    print("Bot(Leader): (uL: {} m/s, omegaL: {} r/s)".format(linearVel, angularVel))
    # Push commands #
    publishVelocities(linearVel, angularVel)

# Stop movement #
stopLeader()

# Petropoulakis Panagiotis
