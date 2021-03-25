import math
import numpy as np
class Point2D:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.neighbours=[]
        self.permutations=[]

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

class Cluster:
    def __init__(self,points=None):
        if points is None:
            self.points = []
        elif not isinstance(points,list):
            print("WARNING: Cluster accepts a list of 'Point2D' objects")
        else:
            self.points=points
        self.permutations=[]

    def __getitem__(self, item):
        return self.points[item]

    def addPoint(self,point):
        if not isinstance(point,Point2D):
            print("WARNING: Cluster.addPoint(point) expects 'point' to be of type 'Point2D'")
            return
        self.points.append(point)

    def getPermutations(self):
        if len(self.permutations) >= math.factorial(len(self.points)):
            return self.permutations
        self.permute(len(self.points), self.points)
        return self.permutations

    def permute(self, k, points):
        if k == 1:
            self.permutations.append(points.copy())
        else:
            for i in range(k):
                self.permute(k - 1, points)
                if (k % 2) == 0:  # if k is even
                    points[0], points[k - 1] = points[k - 1], points[0]  # swap k-1 with 0
                else:
                    points[i], points[k - 1] = points[k - 1], points[i]  # swap k-1 with i


class Pair:
    def __init__(self,point_1,point_2):
        self.point_1=point_1
        self.point_2=point_2
    def  __getitem__(self, item):
        if item==0:
            return self.point_1
        elif item==1:
            return self.point_2
    def distance(self):
        distance = self.point_1.distanceTo(self.point_2)
        return distance