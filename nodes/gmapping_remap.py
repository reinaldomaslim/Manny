#!/usr/bin/env python

import rospy
import math
import numpy as np
from nav_msgs.msg import Odometry, OccupancyGrid
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class MapRepub(object):

    costmap=OccupancyGrid()
    costmap_final=OccupancyGrid()

    def __init__(self):
        #this node inflates walls from gmapping for movebase extra margin
        rospy.init_node('gmapping_remap', anonymous=False)

        self.map_pub =rospy.Publisher("/map", OccupancyGrid, queue_size=10)

        #Wait for map and start frontier explorer navigation
        self.map_received = False
        rospy.wait_for_message("/map/raw", OccupancyGrid)
        rospy.Subscriber("/map/raw", OccupancyGrid, self.map_callback, queue_size = 50)
        while not self.map_received:
            rospy.sleep(1)
        print("map received")

        rate=rospy.Rate(3)

        while not rospy.is_shutdown():
            
            self.map_pub.publish(self.costmap_final)
            rate.sleep()


    def map_callback(self, msg):

        #append offset points from walls
        self.map_received=True

        self.map_width=msg.info.width
        self.map_height=msg.info.height 
        self.map_resolution=msg.info.resolution
        self.origin=msg.info.origin
        #create visitancy map

        #mark inspected walls 
        self.costmap.data=np.copy(msg.data)
        self.createCostmap(20, msg.data)
        self.costmap.info=msg.info
        self.costmap.header=msg.header           

        self.costmap_final=self.costmap


    def createCostmap(self, radius, map_data):
        #costmap needed to constrain robot from getting too close to the wall

        for point in range(len(map_data)):
            if map_data[point]>0:
                self.updateGrid(point, radius)

    def updateGrid(self, point, radius):

        for i in range(-int(radius/2), int(radius/2)):
            for j in range(-int(radius/2), int(radius/2)):
                self.costmap.data[point+i+j*self.map_width]=100



if __name__ == '__main__':
    try:
        map_remap=MapRepub()
    except rospy.ROSInterruptException:
        rospy.loginfo("Finished")
