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

    def transformPoint(self,tx,ty,theta):
        p = np.array([[self.x],[self.y],[1]]) #homogeneous representation of point
        A = [[np.cos(theta), -np.sin(theta),tx],[np.sin(theta),np.cos(theta),ty],[0,0,1]] #homogeneous transform
        p_prime = np.matmul(A,p)
        return (Point2D(p_prime[0][0],p_prime[1][0]))

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

    def transformCluster(self,tx,ty,theta):
        points=[]
        for p in self.points:
            points.append(p.transformPoint(tx,ty,theta))
        return Cluster(points)

class LocalCluster:
    def __init__(self,origin,threshold=2.0,neighbours=None):
        if not isinstance(origin,Point2D):
            print("WARNING: Expected LocalCluster(origin) argument 'origin' to be of type 'Point2D'")
        self.origin = origin
        if neighbours is not None:
            self.neighbours=neighbours
        else:
            self.neighbours = []
        self.threshold=threshold

    def __getitem__(self, item):
        if item==0:
            return self.origin
        else:
            return self.neighbours[item]

    def getNumberOfNeigbours(self):
        return len(self.neighbours)
    def addNeighbour(self,point):
        if not isinstance(point,Point2D):
            print("WARNING: Attempted to add neighbor to LocalCluster which is not of type Point2D.")
            return
        dx = point[0]-self.origin[0]
        dy = point[1]-self.origin[1]
        self.neighbours.append(Point2D(dx,dy))

    def rotateLocalCluster(self,theta):
        neighbours = []
        for p in self.neighbours:
            neighbours.append(p.transformPoint(0,0,theta))
        return LocalCluster(self.origin,neighbours=neighbours,threshold=self.threshold)

    def compareDistance(self,cluster):
        used_idx=[]
        matched_points=[]
        ranges=[]
        for i in range(len(self.neighbours)):
            shortest_dist=np.inf
            shortest_idx = None
            for j in range(len(cluster.neighbours)):
                if j in used_idx: #cannot reuse the same point
                    continue
                else:
                    dist=self.neighbours[i].distanceTo(cluster.neighbours[j])
                    #dist=angle(self.origin,self.neighbours[i],cluster.neighbours[j])
                    if dist < self.threshold and dist < shortest_dist:
                        shortest_dist=dist
                        shortest_idx=j
            if not shortest_dist==np.inf:
                used_idx.append(shortest_idx)
                ranges.append(shortest_dist)
                matched_points.append([i,shortest_idx])

        if len(ranges) == 0:
            return (np.inf,0,matched_points)
        distance = 0
        numberOfMatches=0
        for i in range(len(ranges)):
            if ranges[i]<self.threshold:
                distance += ranges[i]*ranges[i]
                numberOfMatches+=1
        distance=np.sqrt(distance)/numberOfMatches
        return(distance,numberOfMatches,matched_points)


def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))
def length(v):
  return math.sqrt(dotproduct(v, v))
def angle(origin,p1,p2):
    v1 = [p1[0] - origin[0], p1[1] - origin[1]]
    v2 = [p2[0] - origin[0], p2[1] - origin[1]]
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


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
    def transformPair(self,tx,ty,theta):
        p1 = self.point_1.transformPoint(tx,ty,theta)
        p2 = self.point_2.transformPoint(tx,ty,theta)
        return Pair(p1,p2)