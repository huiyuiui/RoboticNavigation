import cv2
import sys
sys.path.append("..")
import PathPlanning.utils as utils
from PathPlanning.planner import Planner
from queue import PriorityQueue

class PlannerAStar(Planner):
    def __init__(self, m, inter=10):
        super().__init__(m)
        self.inter = inter
        self.initialize()

    def initialize(self):
        self.queue = PriorityQueue()
        self.parent = {}
        self.h = {} # Distance from start to node
        self.g = {} # Distance from node to goal
        self.f = {} # Total cost
        self.visited = []
        self.inqueue = []
        self.goal_node = None

    def planning(self, start=(100,200), goal=(375,520), inter=None, img=None):
        if inter is None:
            inter = self.inter
        start = (int(start[0]), int(start[1]))
        goal = (int(goal[0]), int(goal[1]))
        # Initialize 
        self.initialize()
        self.parent[start] = None
        self.g[start] = 0
        self.h[start] = utils.distance(start, goal)
        self.f[start] = self.g[start] + self.h[start]
        self.queue.put((start, self.f[start])) # Priority Queue: (node, f(priority))
        self.inqueue.append(start)
        # Main loop
        while(not self.queue.empty()):
            # TODO: A Star Algorithm
            # Get current node
            current_node = self.queue.get()[0]
            # Goal check
            if(utils.distance(current_node, goal) <= inter):
                self.goal_node = current_node
                print("Cost:", self.g[current_node])
                break
            # Add to visited
            self.visited.append(current_node)
            # Relaxation
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if(x == 0 and y == 0):
                        continue
                    next_node = (current_node[0] + x * inter, current_node[1] + y * inter)
                    # Boundary check
                    if(next_node[0] < 0 or next_node[0] >= self.map.shape[1] or next_node[1] < 0 or next_node[1] >= self.map.shape[0]):
                        continue
                    # Collision check
                    if(self.map[next_node[1], next_node[0]] < 0.5):
                        continue
                    # Cost calculation
                    next_g = self.g[current_node] + utils.distance(current_node, next_node)
                    next_h = utils.distance(next_node, goal)
                    next_f = next_g + next_h
                    # Visited check
                    if(next_node in self.visited):
                        continue
                    # Update status
                    if(next_node not in self.inqueue or next_g < self.g[next_node]):
                        self.g[next_node] = next_g
                        self.h[next_node] = next_h
                        self.f[next_node] = next_f
                        self.queue.put((next_node, next_f))
                        self.inqueue.append(next_node)
                        self.parent[next_node] = current_node
        
        # Extract path
        path = []
        p = self.goal_node
        if p is None:
            return path
        while(True):
            path.insert(0,p)
            if self.parent[p] is None:
                break
            p = self.parent[p]
        if path[-1] != goal:
            path.append(goal)
        return path
