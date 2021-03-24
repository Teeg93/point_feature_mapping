import math
import numpy as np
class Point2D:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.neighbours=[[]]
    def __getitem__(self, item):
        if item==0:
            return self.x
        elif item==1:
            return self.y
        else:
            print(f"Point2D only accepts index [0] or [1]\nReceived index [{item}]")
            raise Exception

    def distanceTo(self,point):
        if not isinstance(point,Point2D):
            print("WARNING: Point2D.distanceTo(point) expects 'point' to be of type 'Point2D'")
            return
        dx = point[0]-self[0]
        dy = point[1]-self[1]
        dist = math.sqrt(dx*dx + dy*dy)
        return dist

    def addNeighbour(self,point,dist):
        if not isinstance(point,Point2D):
            print("WARNING: Point2D.distanceTo(point) expects 'point' to be of type 'Point2D'")
            return
        self.neighbours.append([point,dist])

    def getNearestNeighobur(self):
        pass

