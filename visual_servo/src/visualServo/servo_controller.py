import rospy
from message_filters import Subscriber
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from scipy.spatial.transform import Rotation as R
import numpy as np
import signal
import math
import time
import helpers
import copy

class Controller():

    def __init__(self):
        rospy.init_node('servo_controller', disable_signals=True)

        ##########
        # Topics #
        #########
        self.topicLeader = "/leader/state/"
        self.topicFollower = "/robot/robotnik_base_control/odom/" # This robot 
        self.topicVel = "/robot/robotnik_base_control/cmd_vel/" # This robot

        #############
        # Main vars #
        #############
        self.leaderState = Odometry()
        self.followerState = Odometry()
        self.prevFollowerState = customState() # x, y, theta, u, omega
        self.prevLeaderState = customState()
        self.newLeaderStateFlag = False # New message arrived
        self.newFollowerStateFlag = False

        self.ratePub = 70 # Publish velocity hz
        self.rate = rospy.Rate(self.ratePub)
        self.leaderStateMinRate = 10000  # sec 
        self.followerStateMinRate = 0.3
        self.warmUp = False
        self.debug = True
        self.exitCode = "success"

        ###################
        # Controller vars #
        ###################
        self.servoDistance = 1.7 # Secure distance from leader
        self.eTol = 0.005 # Error tol
        self.eKp = 0.013 # Gain distance
        self.aKp = 0.7 # Gain angle
        self.dKp = 2.2 # Gain line distance

        # Vel ranges #
        self.maxUF = 0.06 # m/s
        self.minUF = 0.025
        self.maxOmegaF = 0.3 # r/s
        self.minOmegaF = 0.01

        # Max - Min permitted ranges of target #
        self.devX = 0.7 # Prev x vs current x of leader
        self.devY = 0.4
        self.maxX = 6.0 # minX == servoDistance 
        self.yDiff = 2.0 # max,min permitted y
        self.thetaDiff = math.pi / 2

        # Set subs #
        self.subLeader = Subscriber(self.topicLeader, Odometry, queue_size=1)
        self.subFollower = Subscriber(self.topicFollower, Odometry, queue_size=1)

        self.subLeader.registerCallback(self.readLeaderState)
        self.subFollower.registerCallback(self.readFollowerState)

        # Set pub #
        self.pubVel = rospy.Publisher(self.topicVel, Twist, queue_size=1)

        # Set sig handler for proper termination #
        signal.signal(signal.SIGINT, self.sigHandler)
        signal.signal(signal.SIGTERM, self.sigHandler)
        signal.signal(signal.SIGTSTP, self.sigHandler)

        if(self.debug):
            self.__str__()

    # Leader callback #
    def readLeaderState(self, odom):
        self.leaderState = odom
        self.newLeaderStateFlag = True

    # Follower callback #
    def readFollowerState(self, odom):
        self.followerState.pose = odom.pose
        self.newFollowerStateFlag = True

    # Main loop     #
    # Follow leader #
    def start(self):

        # Initialize current states #
        leaderState = customState()
        followerState = customState()

        try:
            while not rospy.is_shutdown():

                # Wait messages #
                if self.warmUp == False and (self.newLeaderStateFlag == False or self.newFollowerStateFlag == False):
                    continue

                # New leader state arrived #
                if self.newLeaderStateFlag:
                    self.newLeaderStateFlag = False

                    # Get state #
                    leaderState = self.leaderState
                    startLeaderTimer = rospy.get_time() # Start timer
                    leaderState = customState.fixState(leaderState) # Fix state

                    # Check if we lost leader #
                    if self.lostLeader(leaderState, followerState):
                        self.stopFollower()
                        break

                    # Track states #
                    self.prevLeaderState = copy.deepcopy(leaderState)

                    # Read and save current follower state    #
                    # prevFollowerState = odom of follower at #
                    # the moment of a new leader state        #
                    if self.newFollowerStateFlag:
                        self.newFollowerStateFlag = False
                        followerState = self.followerState
                        startFollowerTimer = rospy.get_time() # Start timer
                        followerState = customState.fixState(followerState) # Fix state
                        self.prevFollowerState = copy.deepcopy(followerState)
                    else:
                        elapsed = rospy.get_time() - startFollowerTimer
                        if elapsed > self.followerStateMinRate:
                            self.stopFollower()
                            self.exitCode = "Follower topic died"
                            break

                    # Fix leader pos #
                    xError = leaderState.x # For safety
                    yError = leaderState.y
                    if xError != 0 or yError != 0:
                        thetaL = math.atan2(yError, xError)

                    # Calculate line #
                    aLine = xError - self.servoDistance
                    bLine = -yError

                # Leader message is missing #
                else:
                    elapsed = rospy.get_time() - startLeaderTimer
                    if elapsed > self.leaderStateMinRate:
                        self.stopFollower()
                        self.exitCode = "Leader topic died"
                        break
                    else:

                        # Read follower state #
                        if self.newFollowerStateFlag:
                            self.newFollowerStateFlag = False
                            followerState = self.followerState
                            startFollowerTimer = rospy.get_time() # Start timer
                            followerState = customState.fixState(followerState)
                        else:
                            elapsed = rospy.get_time() - startFollowerTimer
                            if elapsed > self.followerStateMinRate:
                                self.stopFollower()
                                self.exitCode = "Follower topic died"
                                break

                        # Use previous state of leader as current, use the odometry of follower and fix current error #
                        xDiffOdom = followerState.x - self.prevFollowerState.x
                        yDiffOdom = followerState.y - self.prevFollowerState.y

                        # Fix leader pos - orientation is the same #
                        xError = leaderState.x - xDiffOdom
                        yError = leaderState.y - yDiffOdom

                        # Fix prev leader state #
                        self.prevLeaderState.x = leaderState.x - xDiffOdom
                        self.prevLeaderState.y = leaderState.y - yDiffOdom

                ##############
                # Fix errors #
                ##############
                xNew = leaderState.x - xError # Find coordinates of current point 
                yNew = leaderState.y - yError

                d = abs(aLine*yNew + bLine*xNew) / math.sqrt(leaderState.x**2 + leaderState.y ** 2) # Distance from line 
                e = math.sqrt((xError ** 2) + (yError ** 2)) - self.servoDistance # Distance from goal
                a = thetaL - followerState.theta # Angle goal 

                if self.debug:
                    print('Errors(a: {} d: {} e: {} xError: {} yError: {})'.format(math.degrees(a), d, e, xError - self.servoDistance, yError))

                ###############################
                # Calculate follower commands #
                ###############################
                omegaF = self.aKp * a + self.dKp * d

                if e > self.eTol and xError >= 0:
                    uF = self.eKp * e
                else:
                    uF = 0

                # Push commands #
                self.publishVelocities(uF, omegaF)

                if not self.warmUp:
                    self.warmUp = True

            print("Exit code: " + self.exitCode)
            print("Warm up: " + str(self.warmUp))

        except Exception, e: # Exception occurred 
            print("Exception occurred:  " + str(e))

            # Stop robot #
            self.stopFollower()

            # Stop ros #
            rospy.signal_shutdown(0)

            print("Exit code: " + self.exitCode)
            print("Warm up: " + str(self.warmUp))
            exit()

    # Check states of robots #
    def lostLeader(self, leaderState, followerState):

        if self.warmUp == True and abs(leaderState.x - self.prevLeaderState.x) > self.devX:
            self.exitCode = "lost target: dev x: " + str(leaderState.x) + ", " + str(self.prevLeaderState.x)
            return True

        if self.warmUp == True and abs(leaderState.y - self.prevLeaderState.y) > self.devY:
            self.exitCode = "lost target: dev y: " + str(leaderState.y) + ", " + str(self.prevLeaderState.y)
            return True

        if leaderState.x > self.maxX or leaderState.x < self.servoDistance:
            self.exitCode = "lost target x: " + str(leaderState.x)
            return True

        d = math.sqrt((leaderState.x ** 2) + (leaderState.y ** 2))
        if d < self.servoDistance:
            self.exitCode = "lost target distance: " + str(d)
            return True

        if abs(leaderState.y) > self.yDiff:
            self.exitCode = "lost target y: " + str(leaderState.y)
            return True

        xError = leaderState.x
        yError = leaderState.y
        if xError != 0 or yError != 0:
            thetaL = math.atan2(yError, xError - self.servoDistance)

            if abs(thetaL - followerState.theta) >= self.thetaDiff:
                self.exitCode = "lost target theta: " + str(math.degrees(thetaL)) + ", " + str(followerState.theta)
                return True

        return False

    # Publish velocities to robot topic #
    def publishVelocities(self, u, omega):
        velMsg = Twist()

        # Normalize velocities if needed #
        u, omega = helpers.normalizeVelocity(u, omega, maxU=self.maxUF, minU=self.minUF, maxOmega=self.maxOmegaF, minOmega=self.minOmegaF)

        velMsg.linear.x = u
        velMsg.linear.y = 0.0
        velMsg.linear.z = 0.0

        velMsg.angular.x = 0.0
        velMsg.angular.y = 0.0
        velMsg.angular.z = omega

        # Publish velocities #
        self.pubVel.publish(velMsg)
        self.rate.sleep()

        if self.debug:
            print('Follower velocity(uF: {} m/s omegaF: {} r/s)'.format(u, omega))

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
        while count < 30:
            print("stop")
            self.pubVel.publish(velMsg)
            count += 1

    # Handle signals for proper termination #
    def sigHandler(self, num, frame):

        print("Signal occurred:  " + str(num))

        # Stop robot #
        self.stopFollower()

        # Stop ros #
        rospy.signal_shutdown(0)

        print("Exit code: " + self.exitCode)
        print("Warm up: " + str(self.warmUp))
        exit()

    # Info about visual servo controller #      
    def __str__(self):
        print("Controller parameters:")
        print("======================================================")
        print("Publish max rate: " + str(self.ratePub))
        print("Leader state min rate: " + str(self.leaderStateMinRate))
        print("Follower state min rate: " + str(self.followerStateMinRate))
        print("Goal distance from follower: " + str(self.servoDistance))
        print("Error tolerance: " + str(self.eTol))
        print('Gains(aKp: {} eKp: {} dKp)'.format(self.eKp, self.aKp, self.dKp))
        print("Permitted max follower linear velocity: " + str(self.maxUF))
        print("Permitted min follower linear velocity: " + str(self.minUF))
        print("Permitted max follower angular velocity: " + str(self.maxOmegaF))
        print("Permitted min follower angular velocity: " + str(self.minOmegaF))
        print("Permitted deviation of x: " + str(self.devX))
        print("Permitted deviation of y: " + str(self.devY))
        print("Max permitted x: " + str(self.maxX))
        print("Permitted difference of y: " + str(self.yDiff))
        print("Permitted difference of theta: " + str(math.degrees(self.thetaDiff)))

# Odom to customState #
class customState():
    def __init__(self, x=0, y=0, theta=0, u=0, omega=0):
        self.x = x
        self.y = y
        self.theta = theta
        self.u = u
        self.omega = omega

    @staticmethod # State: Odometry() ros 
    def fixState(state):
        quat = [state.pose.pose.orientation.x,
                state.pose.pose.orientation.y,
                state.pose.pose.orientation.z,
                state.pose.pose.orientation.w]

        # Get heading #
        if all(x == 0 for x in quat):
            theta = 0
        else:
            r = R.from_quat(quat)
            theta = r.as_euler('zyx', degrees=False)[0]

        x = state.pose.pose.position.x
        y = state.pose.pose.position.y
        u = state.twist.twist.linear.x
        omega = state.twist.twist.angular.z

        return customState(x, y, theta, u, omega)

if __name__ == "__main__":

    # Init #
    robotController = Controller()

    # Calculate and publish robot velocities #
    robotController.start()

# Petropoulakis Panagiotis
