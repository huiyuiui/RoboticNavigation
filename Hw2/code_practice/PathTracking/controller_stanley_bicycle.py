import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerStanleyBicycle(Controller):
    def __init__(self, kp=0.5):
        self.path = None
        self.kp = kp

    # State: [x, y, yaw, delta, v, l]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None
        
        # Extract State 
        x, y, yaw, delta, v, l = info["x"], info["y"], info["yaw"], info["delta"], info["v"], info["l"]

        # Search Front Wheel Target
        front_x = x + l*np.cos(np.deg2rad(yaw))
        front_y = y + l*np.sin(np.deg2rad(yaw))
        vf = v / np.cos(np.deg2rad(delta))
        min_idx, min_dist = utils.search_nearest(self.path, (front_x,front_y))
        target = self.path[min_idx]

        # TODO: Stanley Control for Bicycle Kinematic Model
        # if the target is the last node in the path, return the current delta
        if(min_idx+1 == len(self.path)):
            return delta, target
        # calculate angle error
        next_node = self.path[min_idx+1]
        theta_p = np.arctan2(next_node[1] - target[1], next_node[0] - target[0])
        theta_e = theta_p - np.deg2rad(yaw)
        theta_e = (theta_e + np.pi) % (2 * np.pi) - np.pi # normalize to [-pi, pi]

        # calculate cross track error
        e = (target[0] - x) * np.cos(theta_p + np.pi / 2) + (target[1] - y) * np.sin(theta_p + np.pi / 2)
        next_delta = np.arctan2(self.kp * e, vf) + theta_e

        return np.rad2deg(next_delta), target
