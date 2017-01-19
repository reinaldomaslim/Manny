#!/usr/bin/env python

""" Reinaldo
    3-11-16

"""
import rospy
import actionlib
from actionlib_msgs.msg import *
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from sensor_msgs.msg import RegionOfInterest, CameraInfo, LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from visualization_msgs.msg import Marker
from math import radians, pi, sin, cos, tan, ceil, atan2, sqrt

class FollowWall(object):

    currentScan=LaserScan()


    def __init__(self, nodename):
        rospy.init_node(nodename, anonymous=False)
	
        rospy.on_shutdown(self.shutdown)
    
        rospy.Subscriber("/odometry/filtered", Odometry, self.odom_callback, queue_size = 50)

        rospy.Subscriber("/scan", LaserScan, self.scan_callback, queue_size = 50)

        # Publisher to manually control the robot (e.g. to stop it, queue_size=5)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=5)
	
        self.side=rospy.get_param("~follow_wall/isRight", True)
        self.top_set=rospy.get_param("~follow_wall/top_subset", 6)
        self.bottom_set=rospy.get_param("~follow_wall/bottom_subset", 3)
	
        rate=rospy.Rate(10)
	
        while not rospy.is_shutdown():
            self.top_list=self.createTopSet()
            rospy.loginfo(self.top_list)
            #self.bottom_list=self.createBottomSet()
            #rospy.loginfo(self.bottom_list) 
            error=self.compute_benchmark()
            rospy.loginfo(error)
            rate.sleep()


    def createTopSet(self):
        result=list()
	
        increment=int(pi/(2*self.top_set*self.currentScan.angle_increment))
        for i in range(self.top_set):
            if self.side:
                result.append(self.interpolate(90+i*increment, 5))
            else:
                result.append(self.interpolate(450-i*increment, 5))

        return result

    def createBottomSet(self):
        result=list()
        increment=int(pi/(4*self.bottom_set*self.currentScan.angle_increment))
        for i in range(self.bottom_set):
            if self.side:
                result.append(self.interpolate(90-(i+1)*increment, 5))
            else:
                result.append(self.interpolate(450+(i+1)*increment, 5))	
        return result

    def scan_callback(self, msg):

        self.currentScan=msg
	
    def interpolate(self, index, N):
	
        data=list()
	
        for i in range(N):
            data.append(self.currentScan.ranges[index-int(N/2)+i])

        return self.mean(self.median_filter(data)) 

    def compute_benchmark(self):
        result=list()
	
        d=self.top_list[0]
        increment=pi/(2*self.top_set)

        for i in range(1, self.top_set):
            result.append(d/cos(i*increment)-self.top_list[i])
        return sum(result)
		

    def filter_distance(self, msg):
        for i in range(len(msg.data)):
            if msg.ranges[i] < msg.range_min or msg.ranges[i] > msg.range_max:
            msg.ranges[i]=0
        return msg 

    def median(self, data):

        data = sorted(data)
        n = len(data)

        if n == 0:
            raise StatisticsError("no median for empty data")
        if n%2 == 1:
            return data[n//2]
        else:
            i = n//2
            return (data[i - 1] + data[i])/2

    def mean(self, numbers):
        return float(sum(numbers)) / max(len(numbers), 1)

    def median_filter(self, data):
        result=list()

        for i in range(1, len(data)-1):
            if i==0:
                result.append(self.median([data[i], data[i], data[i+1]]))
            else:
                result.append(self.median([data[i-1], data[i], data[i+1]]))
	
        result.append(self.median([data[len(data)-2], data[len(data)-1]]))
        return result

    def convert_relative_to_absolute(self, boat, target):
        """ boat is catersian (x0, y0),
        target is polar (r, theta)
        and absolute is catersian (x1, y1) """
        r, theta = target
        x, y, yaw = boat
        heading = theta + (yaw - pi / 2)
        center = [x + r * cos(heading),
                  y + r * sin(heading),
                  0]

        return [center, heading]

    def move(self, goal, mode, mode_param):
	  
        finished_within_time = True
	    go_to_next= False
           
  	    if mode==1: #continuous movement function, mode_param is the distance from goal that will set the next goal
		while sqrt((self.x0-goal.target_pose.pose.position.x)**2+(self.y0-goal.target_pose.pose.position.y)**2)>mode_param:
		    rospy.sleep(rospy.Duration(1))
		go_to_next=True

	    elif mode==2: #stop and rotate mode, mode_param is rotational angle in rad
		self.rotation(mode_param)
		self.rotation(-2*mode_param)
		self.rotation(mode_param)
		
    def odom_callback(self, msg):
        """ call back to subscribe, get odometry data:
        pose and orientation of the current boat,
        suffix 0 is for origin """
        self.x0 = msg.pose.pose.position.x
        self.y0 = msg.pose.pose.position.y
        self.z0 = msg.pose.pose.position.z
        x = msg.pose.pose.orientation.x
        y = msg.pose.pose.orientation.y
        z = msg.pose.pose.orientation.z
        w = msg.pose.pose.orientation.w
        self.roll0, self.pitch0, self.yaw0 = euler_from_quaternion((x, y, z, w))
        self.odom_received = True
        # rospy.loginfo([self.x0, self.y0, self.z0])

    def rotation(self, ang):

        rate = rospy.Rate(10)
        an_vel=0.2
        duration=ang/an_vel;
        msg=Twist(Vector3(0.0, 0.0, 0.0), Vector3(0.0, 0.0, an_vel))

        start_time = rospy.get_time()

        while not rospy.is_shutdown():
            current_time=rospy.get_time()
            if (current_time - start_time) > duration:
                self.cmd_vel_pub.publish(Twist(Vector3(0, 0.0, 0.0), Vector3(0.0, 0.0, -2*an_vel)))
                rospy.sleep(0.3)
                self.cmd_vel_pub.publish(Twist())
                break
            self.cmd_vel_pub.publish(msg)
            rate.sleep()

    def shutdown(self):
        rospy.loginfo("Stopping the robot...")
        rospy.sleep(2)
        # Stop the robot
        self.cmd_vel_pub.publish(Twist())
        rospy.sleep(1)

if __name__ == '__main__':
    try:
        FollowWall(nodename="follow_wall")
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")
