#!/usr/bin/env python

import rospy
import math
import time
import numpy as np
import random
import actionlib

from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.preprocessing import StandardScaler, normalize
from collections import Counter

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from visualization_msgs.msg import MarkerArray, Marker
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry, OccupancyGrid
from std_msgs.msg import Int8

import cv2

class WallScanExplorer(object):
    x0, y0, yaw0 = 0, 0, 0
    frontier_distance=list()
    map_width, map_height, map_resolution=0, 0, 0
    distance_to_wall=2
    n_points=12 #decrease this to increase number of inspection points
    n_points_secondary=5
    isLeft=True #thermal camera on the left, wall always on the left. go clockwise inner path
    cluster_centers=list()
    use_costmap=False
    theshold_distance=10
    visited_costmap=OccupancyGrid()
    inspected_costmap=OccupancyGrid()
    costmap=OccupancyGrid()
    next_frontier=None
    threshold_ratio=2.5
    total_distance_travelled=0
    inspection_pose=Odometry()

    def __init__(self):
        
        print("starting wall exploration")
        rospy.init_node('wall_scan_explorer', anonymous=True)
        self.init_markers_frontiers()
        self.init_markers_centers()
        self.init_frontiers()
        self.cmd_vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        #wait for odom
        self.odom_received = False
        rospy.wait_for_message("/odometry/filtered", Odometry)
        rospy.Subscriber("/odometry/filtered", Odometry, self.odom_callback, queue_size=50)
        while not self.odom_received:
            rospy.sleep(1)
        print("odom received")
        
        self.inspection_pose_pub=rospy.Publisher("manny/Inspection", Odometry, queue_size=10)
        # * Subscribe to the move_base action server
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)

        # * Wait 60 seconds for the action server to become available
        rospy.loginfo("Waiting for move_base action server...")
        self.move_base.wait_for_server(rospy.Duration(60))
        rospy.loginfo("Connected to move base server")
        rospy.loginfo("Starting navigation test")

        self.costmap_pub = rospy.Publisher("manny/costmap", OccupancyGrid, queue_size=10)

        self.costmap_received= False
        #Wait for map and start frontier explorer navigation
        self.map_received = False
        self.map_first_callback= True
        rospy.wait_for_message("/map", OccupancyGrid)
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size = 50)
        while not self.map_received:
            rospy.sleep(1)
        print("map received")


        if self.use_costmap:
            rospy.Subscriber("/move_base/global_costmap/costmap", OccupancyGrid, self.costmap_callback, queue_size = 50)

        self.visited_map_pub = rospy.Publisher("visited_map/costmap", OccupancyGrid, queue_size=10)
        self.inspected_map_pub =rospy.Publisher("inspected_map/costmap", OccupancyGrid, queue_size=10)

        self.current_pointOfCompass=None 
        self.init_pos=[self.x0, self.y0]
        inspection_points=0
        while not rospy.is_shutdown():

            if self.map_first_callback:
                rospy.sleep(0.1)
                continue

            idle_pos=[self.x0, self.y0]
            #navigate here, direction of the robot depends on the next target
            next_goal=self.getNextGoal()

            if next_goal is not None:
                self.move_to_goal(next_goal, None)
                #face the correct direction
                self.updateGoals(self.WorldToIndex(next_goal), 1.5) #mark around 1 m of visited goals

                angle=self.correctDirection()
                attempt=1
                while angle is not None and abs(self.yaw0-angle)> 3*math.pi/180:
                    if attempt>5:
                        break
                    print("correcting angle", angle*180/math.pi)
                    self.rotate(angle)
                    angle=self.correctDirection()
                    attempt+=1

                #do inspection here
                print("doing inspection")
                inspection_points+=1
                self.inspection_pose_pub.publish(self.inspection_pose)
                rospy.sleep(0.1)
            else: 
                print("returning to origin")
                self.move_to_goal(self.init_pos, None)
                break

            print("no inspection points:", inspection_points)
            print("total distance: ", self.total_distance_travelled)
            print("wall checked: ", self.computeWallChecked(), "%")
            #if got stuck in false positives wall, use cmd_vel to perform forward escape
            if self.distanceToGoal(idle_pos)<0.2:
                self.forward(0.5)           

            
    def correctDirection(self):
        
        #extract longest line's direction 
        #create 2d numpy array of rolling window diagram
        window_map=self.createWindowMap(3)
        x0, y0=window_map.shape[0]/2, window_map.shape[1]/2
        
        
        #extract lines in rolling window
        rho = 1 # distance resolution in pixels of the Hough grid
        theta = np.pi/180 # angular resolution in radians of the Hough grid
        threshold = 10    # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 10 #minimum number of pixels making up a line
        max_line_gap = 30    # maximum gap in pixels between connectable line segments

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(window_map, rho, theta, threshold, np.array([]), min_line_length, max_line_gap)
        length=0

        if lines is None:
            return None

        for line in lines:
            for x1,y1,x2,y2 in line:
                #find longest line and return its direction
                if math.sqrt((x2-x1)**2+(y2-y1)**2)>length:
                    length=math.sqrt((x2-x1)**2+(y2-y1)**2)
                    alpha=math.atan2(x2-x1, y2-y1)
                    x_mid=(x2+x1)/2
                    y_mid=(y2+y1)/2


        gamma=math.atan2(x_mid-x0, y_mid-y0)

        diff_angle=math.atan2(math.sin(gamma-alpha), math.cos(gamma-alpha))

        if self.isLeft:
            if diff_angle>0:
                return alpha
            else:
                return math.atan2(math.sin(alpha+math.pi), math.cos(alpha+math.pi))
        else:
            if diff_angle<0:
                return alpha
            else:
                return math.atan2(math.sin(alpha+math.pi), math.cos(alpha+math.pi))    

        


    def createWindowMap(self, radius):
        size=int(2*radius/self.map_resolution)
        window_map=np.zeros((size, size), dtype=np.uint8)

        current_index=self.WorldToIndex([self.x0, self.y0])

        for i in range(size):
            for j in range(size):
                index=current_index+(i-int(size/2))+(j-(size/2))*self.map_width
                if index<0 or index>self.map_width*self.map_height:
                    continue

                if self.map_data[index]>0:
                    window_map[i][j]=1
                    self.inspected_map[index]=100

        self.inspected_costmap.data=self.inspected_map
        self.inspected_map_pub.publish(self.inspected_costmap)
        return window_map

    def getNextGoal(self):
        #based on current location and given point of compass
        #get next goal from self.current_cluster
        isFrontier=False
        next_goal=None

        correct_goal=None
        correct_distance=1000
        correct_cost=1000
        correct_available=False
        alpha=0.4
        #find nearest goal in the right direction
        #get the A* path distance to each cluster centers
        clusters=self.filter_clusters(5)
        print(clusters)
        #convert costmap into grids
        grid=self.costMapGrid()
        #for i in grid:
        #    print(i)

        #convert cluster centers into grid index
        goals=np.asarray(clusters)[:, 0]
        #do Djikstra

        #for x in goals:
        #    print(self.Djikstra(grid, x))
        path=self.search(grid, goals)
        print(path)

        for i in range(len(clusters)):
            center=self.convertToMap(clusters[i])



            if path[i][0]!=0 and self.getCost(path[i], alpha, center)<correct_cost:# and self.isRightDirection(center, direction):
                correct_cost=self.getCost(path[i], alpha, center)
                correct_distance=path[i][0]
                correct_goal=center
                correct_index=i
                correct_available=True

        if correct_goal is None:
            return None

        print("correct")
        next_goal=correct_goal
        index=correct_index
        distance=correct_distance

        #if next_frontier is closer than next wall_scan goal, visit next_frontier 
        if self.next_frontier is not None and next_goal is not None:
            if distance>self.Djikstra(grid, self.WorldToGrid(self.next_frontier)):
                distance=self.Djikstra(grid, self.WorldToGrid(self.next_frontier))
                print("going to frontier")
                next_goal=self.next_frontier
                isFrontier=True
                self.next_frontier=None

        if next_goal is None:
            distance=self.Djikstra(grid, self.WorldToGrid(self.next_frontier))
            print("going to frontier")
            next_goal=self.next_frontier
            isFrontier=True
            self.next_frontier=None

        if next_goal is not None and not isFrontier:
            self.cluster_centers.remove(clusters[index])

        if next_goal is not None:
            self.total_distance_travelled+=distance

        return next_goal

    def updateGoals(self, point, radius):

        r=radius/self.map_resolution

        for i in range(-int(r/2), int(r/2)):
            for j in range(-int(r/2), int(r/2)):
                self.goals_map[point+i+j*self.map_width]=100

#-------------------------auxiliary functions
    def search(self, grid, goals):
        # ----------------------------------------
        # insert code here
        # ----------------------------------------
        #first print initial value, expand and insert to pool, choose one with least optimal path, if equal to goal return, if cannot expand fail.
        delta = [[0, 1], # go up
             [ -1, 0], # go left
             [ 0, -1], # go down
             [ 1, 0]] # go right
        init=self.WorldToGrid([self.x0, self.y0])
        cost=1
        pool=[]
        pool.append([])
        pool[0].append([0, int(init[0]), int(init[1])])
        visit = grid
        visit[int(init[0])][int(init[1])]=1
        index=1
        path=np.zeros((len(goals), 3))
        found=0




        while(len(pool)!=0):
            current=[]
            
            current=pool.pop(0)
            index-=1

            new=[]
            for i in range(len(delta)):
                new=[current[0][0]+cost, current[0][1]+delta[i][0], current[0][2]+delta[i][1]]

                if new[1]<0 or new[1]>len(grid)-1 or new[2]<0 or new[2]>len(grid[0])-1:
                    continue

                if visit[new[1]][new[2]]==0 and grid[new[1]][new[2]]==0:
                    pool.append([])
                    pool[index].append(new)
                    index+=1
                    visit[new[1]][new[2]]=1

            for i in range(len(goals)): 
                if current[0][1]==int(goals[i][0]) and current[0][2]==int(goals[i][1]):
                    path[i]=current[0]
                    found+=1

            if found==len(goals):
                break

        for i in range(len(path)):
            if path[i][0]==0:
                path[i][0]=abs(init[0]-int(goals[i][0]))+abs(init[1]-int(goals[i][1]))
        path[:, 0]=path[:, 0]*self.map_resolution
        print("found:", found)
        return path

    def Djikstra(self, grid, goal):
        # ----------------------------------------
        # insert code here
        # ----------------------------------------
        #first print initial value, expand and insert to pool, choose one with least optimal path, if equal to goal return, if cannot expand fail.
        delta = [[0, 1], # go up
             [ -1, 0], # go left
             [ 0, -1], # go down
             [ 1, 0]] # go right
        init=self.WorldToGrid([self.x0, self.y0])
        cost=1

        pool=[]
        pool.append([])
        pool[0].append([0, int(init[0]), int(init[1])])
        visit = np.zeros_like(grid)
        visit[int(init[0])][int(init[1])]=1
        index=1
        
        while(len(pool)!=0):

            current=[]
            path=[]
            current=pool.pop(0)
            index-=1
            #print(current)
            if (current[0][1]==goal[0] and current[0][2]==goal[1]):
                path=current[0][0]
                break
                
            new=[]
            for i in range(len(delta)):
                new=[current[0][0]+cost, current[0][1]+delta[i][0], current[0][2]+delta[i][1]]
                
                if new[1]<0 or new[1]>len(grid)-1 or new[2]<0 or new[2]>len(grid[0])-1:
                    continue
                if visit[new[1]][new[2]]==0 and grid[new[1]][new[2]]==0:
                    pool.append([])
                    pool[index].append(new)
                    #print(pool)
                    index+=1
                    visit[new[1]][new[2]]=1
        
        if path==[]:
            path=abs(init[0]-int(goal[0]))+abs(init[1]-int(goal[1]))

        path=path*self.map_resolution 

        return path

    def costMapGrid(self):
        return np.reshape(self.costmap_data, (self.map_height, self.map_width)).T

    def convertClusters(self):
        result=np.zeros((len(self.cluster_centers), 2))
        for i in range(len(self.cluster_centers)): 
            result[i]=self.WorldToGrid(self.cluster_centers[i][0])
        return result

    def getCost(self, path, alpha, goal):
        #(path[i], alpha)
        """
        Return cost for goal selection. Cost=distance+alpha*angle
        angle=difference in goal direction to current direction

        if no preferred reference direction, Cost=distance
        """
        path_length=path[0]

        theta=math.atan2(goal[1]-self.y0, goal[0]-self.x0)

        delta_angle=theta-self.yaw0
        #remap to 0-pi
        delta_angle=abs(math.atan2(math.sin(delta_angle), math.cos(delta_angle)))

        cost=path_length+alpha*delta_angle

        return cost

    def filter_clusters(self, n_goals):
        
        result=list()
        visited=np.zeros(len(self.cluster_centers))
        init=self.WorldToGrid([self.x0, self.y0])
        for i in range(n_goals):
            distance=1000
            for j in range(len(self.cluster_centers)):
                x=self.cluster_centers[j]

                if self.goals_map[self.convertToIndex(x[0])]>0 or visited[j]>0:
                    continue            

                if (abs(init[0]-x[0][0])+abs(init[1]-x[0][1]))<distance:
                    distance=abs(init[0]-x[0][0])+abs(init[1]-x[0][1])
                    nearest=x
                    index=j
            print(distance*self.map_resolution)

            if nearest is None:
                return None

            result.append(nearest)
            visited[index]=1

        return result


    def map_callback(self, msg):

        #go to closest frontier center
        print("map callback")

        if self.map_first_callback:

            #append offset points from walls
            self.map_received=True
            self.map_width=msg.info.width
            self.map_height=msg.info.height 
            self.map_resolution=msg.info.resolution
            self.origin=msg.info.origin
            #create visitancy map
             #100 as visited, 0 unvisited 
            self.costmap_received=False

            #mark areas around offset points that has been clustered before, to prevent reclustering
            self.visited_map=np.zeros((self.map_width*self.map_height,), dtype=np.int)
            self.visited_costmap.info=msg.info
            self.visited_costmap.header=msg.header

            #filter out frontiers or offset points so not too close to wall
            self.costmap.info=msg.info
            self.costmap.header=msg.header

            #mark inspected walls 
            self.inspected_map=np.zeros((self.map_width*self.map_height,), dtype=np.int)
            self.inspected_costmap.info=msg.info
            self.inspected_costmap.header=msg.header           

            #mark around visited goals to filter out redundant/close goals
            self.goals_map=np.zeros((self.map_width*self.map_height,), dtype=np.int) 

        self.costmap_radius=int(0.6*self.distance_to_wall/self.map_resolution)

        self.map_data=msg.data
        
        start_time= rospy.get_time()
        if not self.use_costmap:
            self.createCostmap(self.costmap_radius)
        print("making costmap time")
        print(rospy.get_time()-start_time)

        while not self.costmap_received:
            print("waiting for costmap...")
            rospy.sleep(0.1)
    

        start_time= rospy.get_time()
        #find frontiers and wall->update freeOffsetPoints    
        frontiers=self.findFrontiers()
        print("findFrontiers time")
        print(rospy.get_time()-start_time)

        #compute wall scanning points
        start_time= rospy.get_time()
        self.cluster_centers.extend(self.getKmeansCluster(self.freeOffsetPoints))
        #print self.cluster_centers
        print("clusters centers updated")
        print(rospy.get_time()-start_time)


        #find frontier centers
        start_time= rospy.get_time()
        frontier_centers=self.getDbscanCluster(frontiers)
        self.next_frontier=self.getNearestFrontier(frontier_centers)
        print(rospy.get_time()-start_time)


        self.printMarker(self.freeOffsetPoints)
        self.printFrontier(frontier_centers)
        self.printCenter(self.cluster_centers)
        self.visited_costmap.data=self.visited_map
        self.visited_map_pub.publish(self.visited_costmap)

        print("frontier centers updated")
        
        self.info_updated=True

        self.map_first_callback=False



    def createCostmap(self, radius):
        #costmap needed to constrain robot from getting too close to the wall
        self.costmap_data=np.zeros((self.map_width*self.map_height,), dtype=np.int)

        for point in range(len(self.map_data)):
            if self.map_data[point]>0:
                self.updateGrid(point, radius)

        self.costmap_received=True
        self.costmap.data=self.costmap_data
        self.costmap_pub.publish(self.costmap)

    def updateGrid(self, point, radius):

        for i in range(-int(radius/2), int(radius/2)):
            for j in range(-int(radius/2), int(radius/2)):
                self.costmap_data[point+i+j*self.map_width]=100



    def costmap_callback(self, msg):
        print("costmap callback")
        self.costmap_received=True
        self.costmap_data=msg.data


#---wall scan functions---------------------------------------------

    def getKmeansCluster(self, pointsList):


        n_clusters=int(math.ceil(len(pointsList)/self.n_points))
        self.n_points=self.n_points_secondary
        if n_clusters==0:
            return []

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
            
            if self.map_data[self.convertToIndex(position)]!=0:
                #print("continue")
                continue #avoid this cluster position
            
            result.append( [[int(position[0]), int(position[1])]])
            i+=1

        return result


    def getWallOffset(self,  point):
        offsetPoints=list()
        freeOffsetPoints=list()

        #free is 0, unknown is -1, wall is 100
        if self.visited_map[point]==0 and self.isWall(self.map_data, point):
            #set this wall frontier point as visited
            
            self.updateVisitancy(point, 9)
            offsetPoints=self.getOffsetPoints(point)
            
            for offsetPoint in offsetPoints:
                #filter the indexes
                #if offsetPoint[0]<0 or offsetPoint[0]>(self.map_width*self.map_height-1):
                #    offsetPoints.remove(offsetPoint)
                #    continue
                #if adjacent is free, return as a list

                if self.map_data[offsetPoint[0]]==0 and self.costmap_data[offsetPoint[0]]==0:
                    freeOffsetPoints.append(self.convertFromIndex(offsetPoint))

        return freeOffsetPoints

    def updateVisitancy(self, point, radius):

        for i in range(-int(radius/2), int(radius/2)):
            for j in range(-int(radius/2), int(radius/2)):
                self.visited_map[point+i+j*self.map_width]=100

    def getOffsetPoints(self, point):
        delta_index=int(self.distance_to_wall/self.map_resolution)
        offsetPoints=list()


        offsetPoints.append([point-delta_index*self.map_width])
        offsetPoints.append([point-delta_index])
        offsetPoints.append([point+delta_index*self.map_width])
        offsetPoints.append([point+delta_index])

        return offsetPoints

    def isWall(self, map_data, point):
        adjacentPoints=list()
        
        unknownCounter=0
        #free is 0, unknown is -1, wall is 100
        if map_data[point]>0:
            adjacentPoints=list()
            unknownCounter=0
            adjacentPoints=self.getAdjacentPoints(point)
            for adjacentPoint in adjacentPoints:

                if map_data[adjacentPoint]==0:
                    unknownCounter+=1
                if unknownCounter>1:
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
        return [[x, y]]

    def convertToIndex(self, position):
        return int(position[0])+self.map_width*int(position[1])

#---frontier explorer functions----------------------------------------------
    def getNearestFrontier(self, frontier_centers):
        distance_to_goal=1000
        next_frontier=None
        if frontier_centers is None:
            return None

        for center in frontier_centers:
            
            if self.costmap_data[self.WorldToIndex(center[0])]>0:
                #unreachable place based on inflated costmap
                self.n_clusters_-=1
                #print("unreachable")
                #print self.n_clusters_
                continue

            if self.distanceToGoal(center[0])<distance_to_goal:
                next_frontier=center[0]
                #if previous goal is near to this goal, select another goal
                distance_to_goal=self.distanceToGoal(center[0])

        return next_frontier

    def findFrontiers(self):
        frontiers=list()

        self.freeOffsetPoints=list()
        for i in range(len(self.map_data)):
            #update freeOffsetPoints for wall scan
            self.freeOffsetPoints.extend(self.getWallOffset(i))
             #refresh list as empty
            if self.isFrontier(self.map_data, i):
    
                p=Point()
                position=self.IndexToWorld(i)
                p.x=position[0]
                p.y=position[1]
                p.z=0
                frontiers.append(position)

        return frontiers

    def WorldToIndex(self, point):
        #point is in Map x m, y m
        x_origin, y_origin=self.origin.position.x, self.origin.position.y
        _, _, yaw_origin = euler_from_quaternion((self.origin.orientation.x, self.origin.orientation.y, self.origin.orientation.z, self.origin.orientation.w))

        x=((point[0]-x_origin)*math.cos(yaw_origin)+(point[1]-y_origin)*math.sin(yaw_origin))/self.map_resolution
        y=(-(point[0]-x_origin)*math.sin(yaw_origin)+(point[1]-y_origin)*math.cos(yaw_origin))/self.map_resolution

        #print(x, y)
        return int(x)+int(y)*self.map_width

    def IndexToGrid(self, index):
        x=(index%self.map_width)
        y=math.floor(index/self.map_width)
        return [x, y]

    def WorldToGrid(self, point):
        return self.IndexToGrid(self.WorldToIndex(point))   

    def IndexToWorld(self, index):
        x_origin, y_origin=self.origin.position.x, self.origin.position.y
        _, _, yaw_origin = euler_from_quaternion((self.origin.orientation.x, self.origin.orientation.y, self.origin.orientation.z, self.origin.orientation.w))

        x=self.map_resolution*(index%self.map_width)
        y=self.map_resolution*math.floor(index/self.map_width)    

        x_result=x_origin+(x*math.cos(yaw_origin)-y*math.sin(yaw_origin)) 
        y_result=y_origin+(x*math.sin(yaw_origin)+y*math.cos(yaw_origin)) 

        return [x_result, y_result]  

    def getDbscanCluster(self, frontierList):
        
        print("dbscan duration")

        start_time= rospy.get_time()
        frontiers_array=np.asarray(frontierList)

        if len(frontiers_array)==0:
            return None


        X=normalize(frontiers_array)
        #this line takes longest time to run 
        #D = manhattan_distances(frontiers_array, frontiers_array)
        #X = StandardScaler().fit_transform(frontiers_array)
        print(rospy.get_time()-start_time)      
        db = DBSCAN(eps=0.5, min_samples=5).fit(X)
        #db = DBSCAN(eps=0.5, min_samples=5, algorithm='ball_tree', metric='haversine').fit(X)
        print(rospy.get_time()-start_time)
        
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        
         
        # Number of clusters in labels, ignoring noise if present.
        self.n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        clusters = [frontiers_array[labels == i] for i in range(self.n_clusters_)]
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
        
        unknownCounter=0
        #free is 0, unknown is -1, wall is 100
        while not self.costmap_received:
            print("waiting for costmap...")
            rospy.sleep(1)


        if map_data[point]==0 and self.costmap_data[point]==0:
            adjacentPoints=self.getAdjacentPoints(point)
            for adjacentPoint in adjacentPoints:

                if map_data[adjacentPoint]==-1:
                    unknownCounter+=1
                if unknownCounter>1:
                    return True

        return False

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
        
        if markerList is None:
            return
        else:
            for x in markerList:
                #markerList store points wrt 2D world coordinate
                marker=self.convertToMap(x)
                p=Point()

                p.x=marker[0]
                p.y=marker[1]
                p.z=0

                self.markers.points.append(p)
                self.marker_pub.publish(self.markers)

    def printFrontier(self, markerList):
        self.frontiers.points=list()
        
        if markerList is None:
            return

        for x in markerList:
            #markerList store points wrt 2D world coordinate
            
            p=Point()

            p.x=x[0][0]
            p.y=x[0][1]
            p.z=0

            self.frontiers.points.append(p)
            self.frontier_pub.publish(self.frontiers)

    def odom_callback(self, msg):
        self.x0 = msg.pose.pose.position.x
        self.y0 = msg.pose.pose.position.y
        _, _, self.yaw0 = euler_from_quaternion((msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w))
        self.odom_received = True
        self.inspection_pose=msg

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
                if not finished_within_time:
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
    def forward(self, dis):

        rate = rospy.Rate(10)
        vel=0.3
        duration = abs(dis) / vel
        msg = Twist(Vector3(dis*vel/abs(dis), 0.0, 0.0), Vector3(0.0, 0.0, 0.0))

        rate.sleep()
        start_time = rospy.get_time()

        while not rospy.is_shutdown():
            current_time = rospy.get_time()
            if (current_time - start_time) > duration:
                self.cmd_vel_pub.publish(Twist())
                break
            else:
                self.cmd_vel_pub.publish(msg)
            rate.sleep()

    def rotate(self, direction):

        ang=direction-self.yaw0
        ang=math.atan2(math.sin(ang), math.cos(ang))


        rate = rospy.Rate(10)
        an_vel = 0.3
        duration = abs(ang) / an_vel
        msg = Twist(Vector3(0.0, 0.0, 0.0), Vector3(0.0, 0.0, ang*an_vel/abs(ang)))

        rate.sleep()
        start_time = rospy.get_time()

        while not rospy.is_shutdown():
            current_time = rospy.get_time()
            if (current_time - start_time) > duration:
                self.cmd_vel_pub.publish(Twist())
                break
            else:
                self.cmd_vel_pub.publish(msg)
            rate.sleep()

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
        self.center_pub = rospy.Publisher('goal_centers', Marker, queue_size=5)

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


    def init_frontiers(self):
        # Set up our waypoint markers
        marker_scale = 0.2
        marker_lifetime = 0  # 0 is forever
        marker_ns = 'frontiers'
        marker_id = 0
        marker_color = {'r': 0.7, 'g': 0.5, 'b': 1.0, 'a': 1.0}

        # Define a marker publisher.
        self.frontier_pub = rospy.Publisher('frontier_centers', Marker, queue_size=5)

        # Initialize the marker points list.
        self.frontiers = Marker()
        self.frontiers.ns = marker_ns
        self.frontiers.id = marker_id
        # self.markers.type = Marker.ARROW
        self.frontiers.type = Marker.CUBE_LIST
        self.frontiers.action = Marker.ADD
        self.frontiers.lifetime = rospy.Duration(marker_lifetime)
        self.frontiers.scale.x = marker_scale
        self.frontiers.scale.y = marker_scale
        self.frontiers.scale.z = marker_scale
        self.frontiers.color.r = marker_color['r']
        self.frontiers.color.g = marker_color['g']
        self.frontiers.color.b = marker_color['b']
        self.frontiers.color.a = marker_color['a']

        self.frontiers.header.frame_id = 'odom'
        self.frontiers.header.stamp = rospy.Time.now()
        self.frontiers.points = list()


    def computeWallChecked(self):

        nb_checked=np.count_nonzero(self.inspected_costmap.data>0)
        #print("nb_checked", nb_checked)
        nb_wall=np.count_nonzero(np.asarray(self.map_data)>0)
        #print("nb_wall", nb_wall)
        percentage=nb_checked*100/nb_wall
        return percentage



if __name__ == '__main__':
    try:
        wall_scan_explorer=WallScanExplorer()
    except rospy.ROSInterruptException:
        rospy.loginfo("Exploration Finished")
