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
from geometry_msgs.msg import Point, Pose, Quaternion
from visualization_msgs.msg import MarkerArray, Marker
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Int8

class WallScanner(object):
    x0, y0, yaw0 = 0, 0, 0
    frontiers= list()
    frontier_distance=list()
    map_width, map_height, map_resolution=0, 0, 0
    distance_to_wall=1
    n_points=50
    isLeft=True #thermal camera on the left, wall always on the left. go clockwise inner path
    cluster_centers=list()


    def __init__(self):
        
        print("starting wall exploration")
        rospy.init_node('wall_scanner', anonymous=True)
        self.init_markers_frontiers()
        self.init_markers_centers()

        #wait for odom
        self.odom_received = False
        rospy.wait_for_message("/odometry/filtered", Odometry)
        rospy.Subscriber("/odometry/filtered", Odometry, self.odom_callback, queue_size=50)
        while not self.odom_received:
            rospy.sleep(1)
        print("odom received")
        
        # * Subscribe to the move_base action server
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)

        # * Wait 60 seconds for the action server to become available
        rospy.loginfo("Waiting for move_base action server...")
        self.move_base.wait_for_server(rospy.Duration(60))
        rospy.loginfo("Connected to move base server")
        rospy.loginfo("Starting navigation test")

        self.costmap_received= False
        #Wait for map and start frontier explorer navigation
        self.map_received = False
        self.map_first_callback= True
        rospy.wait_for_message("/map", OccupancyGrid)
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size = 50)
        while not self.map_received:
            rospy.sleep(1)
        print("map received")



        #rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid, self.costmap_callback, queue_size = 50)

        self.current_pointOfCompass=None 
        self.init_pos=[self.x0, self.y0]

        while not rospy.is_shutdown():

            if self.map_first_callback:
                rospy.sleep(0.1)
                continue

            #navigate here, direction of the robot depends on the next target
            next_goal=self.getNextGoal()

            if next_goal is not None:
                self.move_to_goal(next_goal, None)
                #do inspection here
            else: 
                self.move_to_goal(self.init_pos, None)
                break

            rospy.sleep(0.1)


    def getNextGoal(self):
        #based on current location and given point of compass
        #get next goal from self.current_cluster
        next_goal=None
        direction=[0, 0]
        if self.current_pointOfCompass is not None:
            if self.current_pointOfCompass==0:
                #east
                direction=[1, 0]
            elif self.current_pointOfCompass==1:
                #north
                direction=[0, -1]
            elif self.current_pointOfCompass==2:
                #west
                direction=[-1, 0]
            elif self.current_pointOfCompass==3:
                #south
                direction=[0, 1]


        distance=1000

        for i in self.cluster_centers:
            center=self.convertToMap(i)

            if math.sqrt((self.x0-center[0])**2+(self.y0-center[1])**2)<distance and self.isRightDirection(center, direction):
                #nearest at the right direction
                distance=math.sqrt((self.x0-center[0])**2+(self.y0-center[1])**2)
                next_goal=center
                index=i
                self.current_pointOfCompass=i[1]


        if next_goal is None:
            #find the nearest one
            distance=1000
            for i in self.cluster_centers:
                center=self.convertToMap(i)

                if math.sqrt((self.x0-center[0])**2+(self.y0-center[1])**2)<distance:
                    #nearest from current position
                    distance=math.sqrt((self.x0-center[0])**2+(self.y0-center[1])**2)
                    next_goal=center
                    index=i
                    self.current_pointOfCompass=i[1]

        if next_goal is not None:
            print("this center is removed")
            self.cluster_centers.remove(index)

        return next_goal

    def isRightDirection(self, goal, direction):

        if direction[0]!=0:
            sign_x=(goal[0]-self.x0)/direction[0]
            return (True if sign_x>0 else False)

        if direction[1]!=0:
            sign_y=(goal[1]-self.y0)/direction[1]
            return (True if sign_y>0 else False)

        return True


    def map_callback(self, msg):

        #go to closest frontier center
        print("map callback")

        if self.map_first_callback:
            self.map_received=True
            self.map_width=msg.info.width
            self.map_height=msg.info.height 
            self.map_resolution=msg.info.resolution
            self.origin=msg.info.origin
            #create visitancy map
            self.visited_map=np.zeros((self.map_width*self.map_height,), dtype=np.int) #1 as visited, 0 unvisited            

        self.map_data=msg.data
        self.frontiers=list()
        self.costmap_received=False

        self.createCostmap(3)

        #while not self.costmap_received:
        #    print("waiting for costmap...")
        #    rospy.sleep(0.1)
        
        freeOffsetPoints=list()

        for i in range(len(msg.data)):
            freeOffsetPoints.extend(self.isWall(i))

        self.printMarker(freeOffsetPoints)

        self.cluster_centers.extend(self.getCluster(freeOffsetPoints))
        print self.cluster_centers
        self.printCenter(self.cluster_centers)
        self.map_first_callback=False

    def createCostmap(self, radius):
        self.costmap_data=np.zeros((self.map_width*self.map_height,), dtype=np.int)

        for point in range(len(self.map_data)):
            if self.map_data[point]==100:
                self.updateGrid(point, radius)


    def updateGrid(self, point, radius):
        adjacentPoints=list()

        for i in range(-int(radius/2), int(radius+2)):
            for j in range(-int(radius/2), int(radius+2)):
                adjacentPoints.append(point+i+j*self.map_width)


        for adjacentPoint in adjacentPoints:
            if adjacentPoint < 0 or adjacentPoint > self.map_height*self.map_width-1:
                continue
            self.costmap_data[adjacentPoint]=100


    def costmap_callback(self, msg):
        print("costmap callback")
        self.costmap_received=True
        self.costmap_data=msg.data

    def getCluster(self, pointsList):
        n_clusters=len(pointsList)/self.n_points

        positionList=list()
        result=list()

        for point in pointsList:
            positionList.append(point[0])

        cluster_array=np.asarray(positionList)
        print cluster_array

        if cluster_array.size==0:
            result=[]
            return result

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
            
            result.append( [[int(position[0]), int(position[1])], direction])
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
        if self.isFrontier(self.costmap_data, point) and self.visited_map[point]==0:
            #set this wall frontier point as visited
            self.visited_map[point]=1

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
            if self.costmap_data[adjacentPoint]>0:
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

    def isFrontier(self, map_data, point):
        adjacentPoints=list()
        adjacentPoints=self.getAdjacentPoints(point)
        unknownCounter=0
        #free is 0, unknown is -1, wall is 100


        if map_data[point]>0:

            for adjacentPoint in adjacentPoints:

                if adjacentPoint<0 or adjacentPoint>(self.map_width*self.map_height-1):
                    adjacentPoints.remove(adjacentPoint)
                    continue

                if map_data[adjacentPoint]==0:
                    unknownCounter+=1
                if unknownCounter>2:
                    return True

        return False

    def getAdjacentPoints(self, point):
        adjacentPoints=list()

        adjacentPoints.append(point-1)
        adjacentPoints.append(point+1)
        adjacentPoints.append(point-1-self.map_width)
        adjacentPoints.append(point-self.map_width)
        adjacentPoints.append(point+1-self.map_width)
        adjacentPoints.append(point-1+self.map_width)
        adjacentPoints.append(point+self.map_width)
        adjacentPoints.append(point+1+self.map_width)

        return adjacentPoints

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

    def move_to_goal(self, goal_position, direction):
        
        if direction is None:
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
        finished_within_time = False
        #finished_within_time = self.move_base.wait_for_result(rospy.Duration(30 * 1))
        start_time= rospy.get_time()
        
        # If we don't get there in time, abort the goal
        duration=60

        while not rospy.is_shutdown():


            current_time = rospy.get_time()
            elapsed=(current_time - start_time)
            #print(elapsed, finished_within_time)

            if elapsed > duration or finished_within_time:
                print("cancelling goal")
                self.move_base.cancel_goal()
                rospy.loginfo("Goal finished, next...")
                break

            finished_within_time=self.move_base.wait_for_result(rospy.Duration(1))
            rospy.sleep(0.1)

        #if not finished_within_time:
        #    self.move_base.cancel_goal()
        #    rospy.loginfo("Goal cancelled, next...")
        #else:
            # We made it!
        #    state = self.move_base.get_state()
        #    if state == 3:
        #        rospy.loginfo("Goal succeeded!")

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
