#!/usr/bin/env python

import rospy
import math
import time
import numpy as np
import random
import actionlib

from frontier_explorer_v2 import FrontierExplorer
from wall_explorer_simple import WallScanExplorer


if __name__ == '__main__':
    try:
    	frontier_explorer=FrontierExplorer()
        wall_scan_explorer=WallScanExplorer()
    except rospy.ROSInterruptException:
        rospy.loginfo("Exploration Finished")