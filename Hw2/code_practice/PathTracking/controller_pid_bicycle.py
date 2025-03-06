import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPIDBicycle(Controller):
    def __init__(self, kp=0.4, ki=0.0001, kd=0.5):
        self.path = None
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.acc_ep = 0
        self.last_ep = 0
    
    def set_path(self, path):
        super().set_path(path)
        self.acc_ep = 0
        self.last_ep = 0
    
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None
        
        # Extract State
        x, y, dt, yaw = info["x"], info["y"], info["dt"], info["yaw"]

        # Search Nesrest Target
        min_idx, min_dist = utils.search_nearest(self.path, (x,y))
        target = self.path[min_idx] # x, y, yaw, curv
        
        # TODO: PID Control for Bicycle Kinematic Model
        # Calculate Error
        et = target[1] - y
        heading_error = target[2] - yaw
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error)) # normalize angle
        self.acc_ep += et * dt
        et_diff = (et - self.last_ep) / dt if dt > 0 else 0 # avoid zero division
        self.last_ep = et
        # PID Control
        P = self.kp * et + 0.5 * heading_error # add heading_error to P
        I = self.ki * self.acc_ep
        D = self.kd * et_diff
        next_delta = P + I + D  
        return next_delta
