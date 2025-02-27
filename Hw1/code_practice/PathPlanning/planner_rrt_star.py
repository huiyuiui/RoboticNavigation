import cv2
import numpy as np
import sys
sys.path.append("..")
import PathPlanning.utils as utils
from PathPlanning.planner import Planner

class PlannerRRTStar(Planner):
    def __init__(self, m, extend_len=20):
        super().__init__(m)
        self.extend_len = extend_len 

    def _random_node(self, goal, shape):
        r = np.random.choice(2,1,p=[0.5,0.5])
        if r==1:
            return (float(goal[0]), float(goal[1]))
        else:
            rx = float(np.random.randint(int(shape[1])))
            ry = float(np.random.randint(int(shape[0])))
            return (rx, ry)

    def _nearest_node(self, samp_node):
        min_dist = 99999
        min_node = None
        for n in self.ntree:
            dist = utils.distance(n, samp_node)
            if dist < min_dist:
                min_dist = dist
                min_node = n
        return min_node

    def _check_collision(self, n1, n2):
        n1_ = utils.pos_int(n1)
        n2_ = utils.pos_int(n2)
        line = utils.Bresenham(n1_[0], n2_[0], n1_[1], n2_[1])
        for pts in line:
            if self.map[int(pts[1]),int(pts[0])]<0.5:
                return True
        return False

    def _steer(self, from_node, to_node, extend_len):
        vect = np.array(to_node) - np.array(from_node)
        v_len = np.hypot(vect[0], vect[1])
        v_theta = np.arctan2(vect[1], vect[0])
        if extend_len > v_len:
            extend_len = v_len
        new_node = (from_node[0]+extend_len*np.cos(v_theta), from_node[1]+extend_len*np.sin(v_theta))
        if new_node[1]<0 or new_node[1]>=self.map.shape[0] or new_node[0]<0 or new_node[0]>=self.map.shape[1] or self._check_collision(from_node, new_node):
            return False, None
        else:        
            return new_node, utils.distance(new_node, from_node)
        
    def _re_parent(self, new_node, near_node, cost, threshold):
        parent_node = near_node
        parent_cost  = self.cost[near_node] + cost # cost from near node to new node
        min_cost = 99999
        for node in self.ntree:
            # if node is near node or distance from node to new node is larger than threshold or collision occurs
            if node == near_node or utils.distance(node, new_node) > threshold or self._check_collision(node, new_node):
                continue
            new_cost = self.cost[node] + utils.distance(node, new_node) # cost from reparent node to new node
            if new_cost < min_cost:
                min_cost = new_cost
                parent_node = node
                parent_cost = new_cost

        return parent_node, parent_cost
    
    def _re_wire(self, new_node, threshold):
        for node in self.ntree:
            edge_cost = utils.distance(node, new_node)
            # if node is near node or distance from node to new node is larger than threshold or collision occurs
            if node == new_node or edge_cost > threshold or self._check_collision(node, new_node):
                continue
            if self.cost[new_node] + edge_cost < self.cost[node]:
                self.ntree[node] = new_node
                self.cost[node] = self.cost[new_node] + edge_cost
                
        return

    def planning(self, start, goal, extend_len=None, img=None):
        if extend_len is None:
            extend_len = self.extend_len
        self.ntree = {}
        self.ntree[start] = None
        self.cost = {}
        self.cost[start] = 0
        goal_node = None
        for it in range(20000):
            #print("\r", it, len(self.ntree), end="")
            samp_node = self._random_node(goal, self.map.shape) # random sample
            near_node = self._nearest_node(samp_node) # find nearest node
            new_node, cost = self._steer(near_node, samp_node, extend_len) # steer
            if new_node is not False:
                # TODO: Re-Parent & Re-Wire
                # Re-Parent
                parent_node, parent_cost = self._re_parent(new_node, near_node, cost, extend_len)
                self.ntree[new_node] = parent_node
                self.cost[new_node] = parent_cost
                # Re-Wire
                self._re_wire(new_node, extend_len)
                # Goal check
                if utils.distance(near_node, goal) < extend_len:
                    goal_node = near_node
                    break
            else:
                continue

            # Draw
            if img is not None:
                for n in self.ntree:
                    if self.ntree[n] is None:
                        continue
                    node = self.ntree[n]
                    cv2.line(img, (int(n[0]), int(n[1])), (int(node[0]), int(node[1])), (0,1,0), 1)
                # Near Node
                img_ = img.copy()
                cv2.circle(img_,utils.pos_int(new_node),5,(0,0.5,1),3)
                # Draw Image
                img_ = cv2.flip(img_,0)
                cv2.imshow("Path Planning",img_)
                k = cv2.waitKey(1)
                if k == 27:
                    break
        
        # Extract Path
        path = []
        n = goal_node
        while(True):
            if n is None:
                break
            path.insert(0,n)
            node = self.ntree[n]
            n = self.ntree[n] 
        path.append(goal)
        return path
