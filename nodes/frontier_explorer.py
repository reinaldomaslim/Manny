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

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion
from visualization_msgs.msg import MarkerArray, Marker
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Int8

class FrontierExplorer(object):
    x0, y0, yaw0 = 0, 0, 0
    frontiers= list()
    frontier_distance=list()
    map_width, map_height, map_resolution=0, 0, 0
    next_goal=[0, 0]
    previous_goal=[0, 0]
    new_goal=False
    terminate=False

    def __init__(self):
        
        print("starting frontier exploration")
        rospy.init_node('frontier_explorer', anonymous=True)
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

        self.costmap_received = False
        #Wait for map and start frontier explorer navigation
        self.map_received = False
        self.map_first_callback= True
        rospy.wait_for_message("/map", OccupancyGrid)
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size = 50)
        while not self.map_received:
            rospy.sleep(1)
        print("map received")

        rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid, self.costmap_callback, queue_size = 50)


        self.init_pos=[self.x0, self.y0]

        while not rospy.is_shutdown():

            idle_pos=[self.x0, self.y0]

            if self.new_goal is True:
                self.move_to_goal(self.next_goal, None)
                self.new_goal=False

            if self.terminate is True:
                #return to initial position
                self.move_to_goal(self.init_pos, None)
                break

            rospy.sleep(0.1)

            #initiate movement for gmapping to register map so that gives map_callback
            if self.distanceToGoal(idle_pos)<0.1:
                self.move_to_goal([self.x0, self.y0], self.yaw0+math.pi/2)

            

    def costmap_callback(self, msg):
        print("costmap callback")
        self.costmap_received=True
        self.costmap_data=msg.data

    def map_callback(self, msg):

        #go to closest frontier center

        print("map callback")

        if self.map_first_callback:
            self.map_received=True
            self.map_width=msg.info.width
            self.map_height=msg.info.height 
            self.map_resolution=msg.info.resolution
            self.origin=msg.info.origin
            self.map_first_callback=False

        self.frontiers=list()

        x_origin, y_origin=self.origin.position.x, self.origin.position.y
        _, _, yaw_origin = euler_from_quaternion((self.origin.orientation.x, self.origin.orientation.y, self.origin.orientation.z, self.origin.orientation.w))

        self.markers.points=list()
        for i in range(len(msg.data)):

             #refresh list as empty
            if self.isFrontier(msg.data, i):
                
                p=Point()

                x=self.map_resolution*(i%self.map_width)
                y=self.map_resolution*math.floor(i/self.map_width)

                p.x=x_origin+(x*math.cos(yaw_origin)-y*math.sin(yaw_origin))
                p.y=y_origin+(x*math.sin(yaw_origin)+y*math.cos(yaw_origin))
                p.z=0

                self.frontiers.append([p.x, p.y])
                self.markers.points.append(p)
                self.marker_pub.publish(self.markers)


        frontier_centers=self.getCluster(self.frontiers)

        distance_to_goal=1000

        self.centers.points=list()
        for center in frontier_centers:
            
            if self.costmap_data[self.convertToIndex(center[0])]>0:
                #unreachable place based on inflated costmap
                self.n_clusters_-=1
                print("unreachable")
                print self.n_clusters_
                continue

            p=Point()
            p.x=center[0][0]
            p.y=center[0][1]
            p.z=0
            self.centers.points.append(p)
            self.center_pub.publish(self.centers)

            if self.distanceToGoal(center[0])<distance_to_goal and self.outsideRadius(center[0], self.previous_goal, 1):
                self.next_goal=center[0]
                #if previous goal is near to this goal, select another goal
                distance_to_goal=self.distanceToGoal(center[0])

        if self.n_clusters_>0:
            self.new_goal=True
            self.previous_goal=self.next_goal
        else:
            self.terminate=True

    def convertToIndex(self, point):
        #point is in Map x m, y m
        x_origin, y_origin=self.origin.position.x, self.origin.position.y
        _, _, yaw_origin = euler_from_quaternion((self.origin.orientation.x, self.origin.orientation.y, self.origin.orientation.z, self.origin.orientation.w))

        x=((point[0]-x_origin)*math.cos(yaw_origin)+(point[1]-y_origin)*math.sin(yaw_origin))/self.map_resolution
        y=(-(point[0]-x_origin)*math.sin(yaw_origin)+(point[1]-y_origin)*math.cos(yaw_origin))/self.map_resolution

        print(x, y)
        return int(x)+int(y)*self.map_width

    def outsideRadius(self, point1, point2, radius):
        if math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)<radius:
            return False
        else:
            return True

    def getCluster(self, frontierList):
        frontiers_array=np.asarray(frontierList)

        db = DBSCAN(eps=0.5, min_samples=15).fit(frontiers_array)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        self.n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print self.n_clusters_

        clusters = [frontiers_array[labels == i] for i in xrange(self.n_clusters_)]
        cluster_centers=list()

        for cluster in clusters:
            frontier_kmeans=KMeans(n_clusters=1).fit(cluster)
            frontier_center=frontier_kmeans.cluster_centers_
            cluster_centers.append(frontier_center)

        return cluster_centers

    def distanceToGoal(self, goal):

        distance=math.sqrt((self.x0-goal[0])**2+(self.y0-goal[1])**2)

        if self.x0==self.init_pos[0] and self.y0==self.init_pos[1]:
            distance=2

        return distance

    def isFrontier(self, map_data, point):
        adjacentPoints=list()
        adjacentPoints=self.getAdjacentPoints(point)
        unknownCounter=0
        #free is 0, unknown is -1, wall is 100
        while not self.costmap_received:
            print("waiting for costmap...")
            rospy.sleep(1)


        if map_data[point]==0:

            for adjacentPoint in adjacentPoints:

                if adjacentPoint<0 or adjacentPoint>(self.map_width*self.map_height-1):
                    adjacentPoints.remove(adjacentPoint)
                    continue

                if map_data[adjacentPoint]==-1:
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
            print(elapsed, finished_within_time)

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
        frontier_exploration=FrontierExplorer()
    except rospy.ROSInterruptException:
        rospy.loginfo("Exploration Finished")
