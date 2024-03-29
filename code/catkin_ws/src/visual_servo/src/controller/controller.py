from __future__ import division
import rospy
from message_filters import Subscriber
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from scipy.spatial.transform import Rotation as R
import numpy as np
import math
import copy
import signal

# Position controller #
class Controller():

    def __init__(self):
        #############
        # Main vars #
        #############
        self.leaderState = State() # x, y, a
        self.prevLeaderState = State()
        self.debug = True
        self.warmUp = False

        ###################
        # Controller vars #
        ###################
        self.servoDistance = 1.7 # Secure distance from leader
        self.eTol = 0.03 # Error tol
        self.eKp = 5.5 # Gain distance
        self.aKp = 7.0 # Gain angle

        # Vel ranges #
        self.maxUF = 0.65 # m/s
        self.maxOmegaF = 2.0 # r/s

        # Max - Min permitted ranges of target #
        self.devX = 0.25
        self.devY = 0.25
        self.maxX = 4 # Initial position of target
        self.yDiff = 2.0
        self.maxServoDistance = math.sqrt(self.maxX ** 2 + self.yDiff ** 2)
        self.aDiff = math.pi / 2

        if(self.debug):
            self.__str__()

    # Main method                                          #
    # x,y goal coordinates must be relative to robot frame # 
    def calculateVelocities(self, x, y):

        exitCode = "Controller: success"

        # Find angle of leader #
        if x != 0 or y != 0:
            a = math.atan2(y, x)
        else:
            return 0, 0, "Controller: bad x, y"

        # Save state #
        self.leaderState = State(x, y, a)

        # Check given target coordinates #
        result, exitCode = self.lostLeader()
        if result:
            return 0, 0, exitCode

        # Track states - for deviation #
        self.prevLeaderState = copy.deepcopy(self.leaderState)

        # Fix errors #
        xError = self.leaderState.x
        yError = self.leaderState.y
        a = self.leaderState.a

        #################
        # Fix distances #
        #################
        eInitG = math.sqrt((xError ** 2) + (yError ** 2)) # Initial distance from goal
        e = eInitG - self.servoDistance # Safe distance from goal  

        # Safe goal coordinates #
        xG = (xError/eInitG) * e
        yG = (yError/eInitG) * e

        if self.debug:
            # a: angle goal 
            # eInit: distance from real goal 
            # e: distance from safe goal 
            # xGoal: safe goal distance x-axis
            # yGoal: safe goal distance y-axis
            # xInit: distance from real goal x-axis
            # yInit: distance from real goal y-axis
            print('Controller: Errors: (a: {} deg, eInit: {} m, e: {} m, xGoal: {} m, yGoal: {} m xInit: {} yInit: {})'.format(math.degrees(a), eInitG, e,  xG, yG, xError, yError))

        ###############################
        # Calculate follower commands #
        ###############################
	if abs(math.degrees(a)) > 1.0:
            omegaF = self.aKp * a
	else:
            omegaF = 0

        # For safety #
	if abs(e) > self.eTol:
            uF = self.eKp * e
        else:
            uF = 0

        # Normalize velocities if needed #
        uF, omegaF = self.normalizeVelocities(uF, omegaF, maxU=self.maxUF, maxOmega=self.maxOmegaF)

        if self.debug:
            print('Controller: Follower velocities: (uF: {} m/s, omegaF: {} r/s)'.format(uF, omegaF))

        if not self.warmUp:
            self.warmUp = True

        return uF, omegaF, exitCode

    # Fix velocities #
    def normalizeVelocities(self, u, omega, maxU=0.1, maxOmega=0.25):

        if(maxU <= 0 or maxOmega <= 0):
            return 0, 0

        if abs(u) > maxU:
            if u > 0:
                u = maxU
            else:
                u = -maxU

        if abs(omega) > maxOmega:
            if omega > 0:
                omega = maxOmega
            else:
                omega = -maxOmega

        return u, omega

    # Check given target coordinates #
    def lostLeader(self):

        # During simulation disable dev constraints #
        if self.warmUp == True and abs(self.leaderState.x - self.prevLeaderState.x) > self.devX:
            exitCode = "Controller: lost target: dev x: " + str(self.leaderState.x) + ", " + str(self.prevLeaderState.x)
            #return True, exitCode

        if self.warmUp == True and abs(self.leaderState.y - self.prevLeaderState.y) > self.devY:
            exitCode = "Controller: lost target: dev y: " + str(self.leaderState.y) + ", " + str(self.prevLeaderState.y)
            #return True, exitCode

        if self.leaderState.x > self.maxX or self.leaderState.x <= 0:
            exitCode = "Controller: lost target: x: " + str(self.leaderState.x)
            return True, exitCode

        d = math.sqrt((self.leaderState.x ** 2) + (self.leaderState.y ** 2))
        if d > self.maxServoDistance:
            exitCode = "Controller: lost target distance: " + str(d)
            return True, exitCode

        if abs(self.leaderState.y) > self.yDiff:
            exitCode = "Controller: lost target y: " + str(self.leaderState.y)
            return True, exitCode

        if abs(self.leaderState.a) >= self.aDiff:
            exitCode = "Controller: lost target a: " + str(math.degrees(self.leaderState.a))
            return True, exitCode

        return False, "Controller: success"

    # Info #
    def __str__(self):
        print("")
        print("Position Controller parameters:")
        print("================================================")
        print("Goal distance from follower: " + str(self.servoDistance)) + " m"
        print("Max distance from follower: " + str(self.maxServoDistance)) + " m"
        print("Error tolerance: " + str(self.eTol)) + " m"
        print('Gains: (aKp: {}, eKp: {})'.format(self.eKp, self.aKp))
        print("Permitted max follower linear velocity: " + str(self.maxUF)) + " m/s"
        print("Permitted max follower angular velocity: " + str(self.maxOmegaF)) + " r/s"
        print("Permitted deviation of x: " + str(self.devX)) + " m"
        print("Permitted deviation of y: " + str(self.devY)) + " m"
        print("Max permitted x: " + str(self.maxX)) + " m"
        print("Permitted difference of y: " + str(self.yDiff)) + " m"
        print("Permitted difference of a: " + str(math.degrees(self.aDiff))) + " deg"
        print("================================================")
        print("")

# Robot state #
class State():
    def __init__(self, x=0, y=0, a=0):
        self.x = x
        self.y = y
        self.a = a

# Gazebo summit-xl experiment - follow leader      #
# Use odometry                                     #
# !! Disable deviation constraint in controller !! #    
class experiment():
    def __init__(self):

        # Call ros #
        rospy.init_node('controller', disable_signals=True)

        self.topicFollower = "/robot/robotnik_base_control/odom/" # Summit-xl 
        self.topicLeader = "/robot_b/robotnik_base_control/odom/" # Summit-xl 
        self.topicVel = "/robot/robotnik_base_control/cmd_vel/"
        self.followerState = Odometry()
        self.leaderState = Odometry()

        # Read state - odom #
        self.subFollower = Subscriber(self.topicFollower, Odometry, queue_size=1)
        self.subFollower.registerCallback(self.readFollowerState)
        self.subLeader = Subscriber(self.topicLeader, Odometry, queue_size=1)
        self.subLeader.registerCallback(self.readLeaderState)

        # Publish velocities commands #
        self.pubVel = rospy.Publisher(self.topicVel, Twist, queue_size=1)

        # Set sig handler for proper termination #
        signal.signal(signal.SIGINT, self.sigHandler)
        signal.signal(signal.SIGTERM, self.sigHandler)
        signal.signal(signal.SIGTSTP, self.sigHandler)

        # Pick controller for testing #
        self.controller = Controller()

    # Callback odom #
    def readFollowerState(self, odom):
        self.followerState.pose = odom.pose

    # Callback odom #
    def readLeaderState(self, odom):
        self.leaderState.pose = odom.pose

    # Odom to x, y with respect to world frame #
    def getPos(self, odom):
        return odom.pose.pose.position.x, odom.pose.pose.position.y

    # Odom to heading #
    def getHeading(self, odom):
        quat = [odom.pose.pose.orientation.x,
                odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z,
                odom.pose.pose.orientation.w]

        # Get heading #
        if all(x == 0 for x in quat):
            theta = 0
        else:
            r = R.from_quat(quat)
            theta = r.as_euler('zyx', degrees=False)[0]

	return theta

    # Conert pos of leader to robot coordinates #
    def worldToRobot(self, goalX, goalY, x, y, theta):
        diff = np.asarray([[goalX - x], [goalY - y], [0]])
        r = R.from_euler("z", -theta, degrees=False)
        r = r.as_dcm()

        result = np.dot(r, diff)
        x = result[0][0]
        y = result[1][0]

        return x, y

    # Main method: Follow the leader #
    def start(self):
        warmUp = False

        while not rospy.is_shutdown():

            # Read pos of leader and follower with respect to world frame #
            followerState = self.followerState
            leaderState = self.leaderState
            x, y = self.getPos(followerState)
            xb, yb = self.getPos(leaderState)
            if(yb == 0 and warmUp == False):
                continue
            elif(yb !=0 and warmUp == False):
                warmUp = True

            # Convert Leaders position relative to the Follower #
            theta = self.getHeading(followerState)
            x, y = self.worldToRobot(xb, yb, x, y, theta)

            # Call controller #
            u, omega, code = self.controller.calculateVelocities(x, y)
            if code != "Controller: success":

                print("Simulation failed")
                print("Error code: " + code)

                # Stop robot #
                self.stopFollower()

                # Close ros #
                rospy.signal_shutdown(0)
                exit()

            # Move follower #
            self.publishVelocities(u, omega)

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
        self.pubVel.publish(velMsg)

    # Stop robot #
    def stopFollower(self):
        velMsg = Twist()

        velMsg.linear.x = 0.0
        velMsg.linear.y = 0.0
        velMsg.linear.z = 0.0

        velMsg.angular.x = 0.0
        velMsg.angular.y = 0.0
        velMsg.angular.z = 0.0

        count = 0
        while count < 30:
            self.pubVel.publish(velMsg)
            count += 1

    # Handle signals for proper termination #
    def sigHandler(self, num, frame):

        print("Signal occurred:  " + str(num))

        # Stop robot #
        self.stopFollower()

        # Close ros #
        rospy.signal_shutdown(0)
        exit()

# Step1: Launch official ros summit-xl simulation with two robot:                #
# roslaunch summit_xl_sim_bringup summit_xl_complete.launch launch_robot_b:=true #
# Step2: Click on a wall and press delete                        		 #
# Step3: Disable deviation constraint in controller             		 #    
# Step4: Run bot.py and wait 1 sec                              		 #
# Step5: Run this module                                        		 #
# Step6: To repeat the simulation terminate bot.py/controller.py and   		 #
# in gazebo press edit and reset the model poses. Then repeat steps 4 and 5      #
if  __name__ == '__main__':

    exp = experiment()
    exp.start()

# Petropoulakis Panagiotis
