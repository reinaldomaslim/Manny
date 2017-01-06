#!/usr/bin/env python

import rospy
import math
import time
import numpy as np
import random

from sklearn.cluster import KMeans
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose
from visualization_msgs.msg import MarkerArray, Marker
from tf.transformations import euler_from_quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Int8

class FrontierExplorer(object):
    x0, y0, yaw0 = 0, 0, 0
    frontiers= list()
    frontier_distance=list()

    def __init__(self):
        print("starting frontier exploration")
        rospy.init_node('frontier_explorer', anonymous=True)

        self.map_received = False
        rospy.wait_for_message("/map", OccupancyGrid)
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size = 50)
        while not self.map_received:
            rospy.sleep(1)
        print("map received")

        while not rospy.is_shutdown():
            rospy.sleep(0.1)


    def map_callback(self, msg):#find white totem
        self.map_received=True
        print msg.info.width
        print msg.info.height



if __name__ == '__main__':
    try:
        test=FrontierExplorer()
    except rospy.ROSInterruptException:
        rospy.loginfo("exploration finished")
