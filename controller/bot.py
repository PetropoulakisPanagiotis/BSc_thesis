from __future__ import division
import rospy
from message_filters import Subscriber
from geometry_msgs.msg import Twist
import numpy as np
import time
import random
import signal

def publishVelocities(pubVel, u, omega):
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
def stopLeader(pubVel):
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

# Call ros #
rospy.init_node('bot', disable_signals=True)

topicVel = "/robot_b/robotnik_base_control/cmd_vel/"

# Publish velocities commands #
pubVel = rospy.Publisher(topicVel, Twist, queue_size=1)

linearVel = 0.5
endTime = time.time() + 60 * 1
angluarTime = time.time() + 10
sig = 1

# Take random actions #
while time.time() < endTime:

    # Update angular velocity #
    if time.time > angluarTime:
        angularVel = random.uniform(0.1, 0.7) * sig
        sig *= -1
        angularTime = time.time()

    # Push commands #
    publishVelocities(pubVel, linearVel, angularVel)

# Stop movement #
stopLeader(pubVel)

# Petropoulakis Panagiotis
