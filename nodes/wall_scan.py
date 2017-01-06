#!/usr/bin/env python

import rospy
import math
import time
import numpy as np
import random
import actionlib

from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from collections import Counter

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose
from visualization_msgs.msg import MarkerArray, Marker
from tf.transformations import euler_from_quaternion
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Int8

class WallScanner(object):
    x0, y0, yaw0 = 0, 0, 0
    frontiers= list()
    frontier_distance=list()
    map_width, map_height, map_resolution=0, 0, 0
    distance_to_wall=0.7
    n_points=30
    isLeft=True #thermal camera on the left, wall always on the left. go clockwise inner path

    def __init__(self):
        
        print("starting wall exploration")
        rospy.init_node('wall_scanner', anonymous=True)
        self.init_markers_frontiers()
        self.init_markers_centers()

        #wait for odom
        #self.odom_received = False
        #rospy.wait_for_message("/odometry/filtered", Odometry)
        #rospy.Subscriber("/odometry/filtered", Odometry, self.odom_callback, queue_size=50)
        #while not self.odom_received:
        #    rospy.sleep(1)
        #print("odom received")
        
        # * Subscribe to the move_base action server
        #self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)

        # * Wait 60 seconds for the action server to become available
        #rospy.loginfo("Waiting for move_base action server...")
        #self.move_base.wait_for_server(rospy.Duration(60))
        #rospy.loginfo("Connected to move base server")
        #rospy.loginfo("Starting navigation test")

        #Wait for map and start frontier explorer navigation
        self.map_received = False
        rospy.wait_for_message("/map", OccupancyGrid)
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size = 50)
        while not self.map_received:
            rospy.sleep(1)
        print("map received")

        #create visitancy map
        self.visited_map=np.zeros((self.map_width*self.map_height,), dtype=np.int)

        while not rospy.is_shutdown():
            #navigate here, direction of the robot depends on the next target
            rospy.sleep(0.1)





    def map_callback(self, msg):

        #go to closest frontier center
        print("map callback")

        self.map_received=True
        self.map_width=msg.info.width
        self.map_height=msg.info.height 
        self.map_resolution=msg.info.resolution
        self.frontiers=list()
        self.origin=msg.info.origin
        self.map_data=msg.data

        
        
        freeOffsetPoints=list()

        for i in range(len(msg.data)):
            freeOffsetPoints.extend(self.isWall(i))

        self.printMarker(freeOffsetPoints)

        cluster_centers=self.getCluster(freeOffsetPoints)
       
        self.printCenter(cluster_centers)



    def getCluster(self, pointsList):
        n_clusters=len(pointsList)/self.n_points

        positionList=list()
        result=list()

        for point in pointsList:
            positionList.append(point[0])

        cluster_array=np.asarray(positionList)
        cluster_kmeans=KMeans(n_clusters=n_clusters).fit(cluster_array)
        cluster_centers=cluster_kmeans.cluster_centers_
        cluster_label=cluster_kmeans.labels_

        i=0
        for position in cluster_centers:
            
            if self.map_data[self.convertToIndex(position)]!=0 or not self.notAroundWall(self.convertToIndex(position)):
                print("continue")
                continue #avoid this cluster position
        
            directionList=list()
            j=0
            for k in cluster_label:
                if k==i:
                    directionList.append(pointsList[j][1])
                j+=1
            
            data=Counter(directionList)
            direction=data.most_common(1)[0][0]
            
            result.append( [[int(position[0]), int(position[1])], data.most_common(1)])
            i+=1

        return result

    def printCenter(self, cluster_centers):
        self.centers.points=list()

        for x in cluster_centers:
            center=self.convertToMap(x)
            p=Point()
            p.x=center[0]
            p.y=center[1]
            p.z=0
            self.centers.points.append(p)
            self.center_pub.publish(self.centers)


    def printMarker(self, markerList):
        self.markers.points=list()
        
        for x in markerList:
            #markerList store points wrt 2D world coordinate
            marker=self.convertToMap(x)
            p=Point()

            p.x=marker[0]
            p.y=marker[1]
            p.z=0

            self.markers.points.append(p)
            self.marker_pub.publish(self.markers)



    def isWall(self,  point):
        offsetPoints=list()
        freeOffsetPoints=list()

        #free is 0, unknown is -1, wall is 100
        if self.map_data[point]==100:
            offsetPoints=self.getOffsetPoints(point)
            
            for offsetPoint in offsetPoints:
                #filter the indexes
                if offsetPoint[0]<0 or offsetPoint[0]>(self.map_width*self.map_height-1):
                    offsetPoints.remove(offsetPoint)
                    continue
                #if adjacent is free, return as a list

                if self.map_data[offsetPoint[0]]==0 and self.notAroundWall(offsetPoint[0]):
                    freeOffsetPoints.append(self.convertFromIndex(offsetPoint))

        return freeOffsetPoints

    def notAroundWall(self, point):

        adjacentPoints=list()

        adjacentPoints.append(point-1)
        adjacentPoints.append(point+1)
        adjacentPoints.append(point-1-self.map_width)
        adjacentPoints.append(point-self.map_width)
        adjacentPoints.append(point+1-self.map_width)
        adjacentPoints.append(point-1+self.map_width)
        adjacentPoints.append(point+self.map_width)
        adjacentPoints.append(point+1+self.map_width)

        for adjacentPoint in adjacentPoints:
            if self.map_data[adjacentPoint]==100:
                return False
        
        return True

    def getOffsetPoints(self, point):
        delta_index=int(self.distance_to_wall/self.map_resolution)
        offsetPoints=list()

        #east=[1, 0]
        #west=[-1, 0]
        #north=[0, -1]
        #south=[0, 1]

        east=0
        north=1
        west=2
        south=3


        offsetPoints.append([point-delta_index*self.map_width, east if self.isLeft else west])
        offsetPoints.append([point-delta_index, north if self.isLeft else south])
        offsetPoints.append([point+delta_index*self.map_width, west if self.isLeft else east])
        offsetPoints.append([point+delta_index, south if self.isLeft else north])

        return offsetPoints

    def convertToMap(self, point2D):
        #point2D is [[x, y], direction]

        #converts occupancyGrid map index into 2D wrt world coordinate
        x_origin, y_origin=self.origin.position.x, self.origin.position.y
        _, _, yaw_origin = euler_from_quaternion((self.origin.orientation.x, self.origin.orientation.y, self.origin.orientation.z, self.origin.orientation.w))
        
        x=self.map_resolution*point2D[0][0]
        y=self.map_resolution*point2D[0][1]
        
        #only return world position for movebase and marker
        return [x_origin+(x*math.cos(yaw_origin)-y*math.sin(yaw_origin)), y_origin+(x*math.sin(yaw_origin)+y*math.cos(yaw_origin))]
        
    def convertFromIndex(self, point):

        x=point[0]%self.map_width
        y=math.floor(point[0]/self.map_width)
        #map in 2D
        return [[x, y], point[1]]

    def convertToIndex(self, position):

        
        return int(position[0])+self.map_width*int(position[1])

    def odom_callback(self, msg):
        self.x0 = msg.pose.pose.position.x
        self.y0 = msg.pose.pose.position.y
        _, _, self.yaw0 = euler_from_quaternion((msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w))
        self.odom_received = True

    def move_to_goal(self, goal_position):
        
        direction=math.atan2(goal_position[1]-self.y0, goal_position[0]-self.x0)
        q_angle = quaternion_from_euler(0, 0, direction)
        q = Quaternion(*q_angle)

        goal = MoveBaseGoal()

        # Use the map frame to define goal poses
        goal.target_pose.header.frame_id = 'map'

        # Set the time stamp to "now"
        goal.target_pose.header.stamp = rospy.Time.now()

        goal.target_pose.pose = Pose(Point(goal_position[0], goal_position[1], 0), q)

        self.move_base.send_goal(goal)
        finished_within_time = True
        self.go_to_next = False
        finished_within_time = self.move_base.wait_for_result(rospy.Duration(60 * 1))

        # If we don't get there in time, abort the goal
        if not finished_within_time or self.go_to_next:
            self.move_base.cancel_goal()
            rospy.loginfo("Goal cancelled, next...")
        else:
            # We made it!
            state = self.move_base.get_state()
            if state == 3: #GoalStatus.SUCCEEDED
                rospy.loginfo("Goal succeeded!")

    def init_markers_frontiers(self):
        # Set up our waypoint markers
        marker_scale = 0.2
        marker_lifetime = 0  # 0 is forever
        marker_ns = 'frontiers'
        marker_id = 0
        marker_color = {'r': 1.0, 'g': 0.7, 'b': 1.0, 'a': 1.0}

        # Define a marker publisher.
        self.marker_pub = rospy.Publisher('frontier_markers', Marker, queue_size=5)

        # Initialize the marker points list.
        self.markers = Marker()
        self.markers.ns = marker_ns
        self.markers.id = marker_id
        # self.markers.type = Marker.ARROW
        self.markers.type = Marker.CUBE_LIST
        self.markers.action = Marker.ADD
        self.markers.lifetime = rospy.Duration(marker_lifetime)
        self.markers.scale.x = marker_scale
        self.markers.scale.y = marker_scale
        self.markers.scale.z = marker_scale
        self.markers.color.r = marker_color['r']
        self.markers.color.g = marker_color['g']
        self.markers.color.b = marker_color['b']
        self.markers.color.a = marker_color['a']

        self.markers.header.frame_id = 'odom'
        self.markers.header.stamp = rospy.Time.now()
        self.markers.points = list()

    def init_markers_centers(self):
        # Set up our waypoint markers
        marker_scale = 0.2
        marker_lifetime = 0  # 0 is forever
        marker_ns = 'frontiers'
        marker_id = 0
        marker_color = {'r': 0.7, 'g': 1.0, 'b': 0.7, 'a': 1.0}

        # Define a marker publisher.
        self.center_pub = rospy.Publisher('frontier_centers', Marker, queue_size=5)

        # Initialize the marker points list.
        self.centers = Marker()
        self.centers.ns = marker_ns
        self.centers.id = marker_id
        # self.markers.type = Marker.ARROW
        self.centers.type = Marker.SPHERE_LIST
        self.centers.action = Marker.ADD
        self.centers.lifetime = rospy.Duration(marker_lifetime)
        self.centers.scale.x = marker_scale
        self.centers.scale.y = marker_scale
        self.centers.scale.z = marker_scale
        self.centers.color.r = marker_color['r']
        self.centers.color.g = marker_color['g']
        self.centers.color.b = marker_color['b']
        self.centers.color.a = marker_color['a']

        self.centers.header.frame_id = 'odom'
        self.centers.header.stamp = rospy.Time.now()
        self.centers.points = list()

if __name__ == '__main__':
    try:
        wall_scanner=WallScanner()
    except rospy.ROSInterruptException:
        rospy.loginfo("Exploration Finished")
