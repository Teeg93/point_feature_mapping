from Point import Point2D
import numpy as np
def four_nearest_neighbours(M,m):
    if not (isinstance(M,list)):
        print(f"WARNING: Expected argument M to be of type 'list', received {type(M)}")
        return
    if not (isinstance(m,Point2D)):
        print(f"WARNING: Expected argument m to be of type 'Point2D', received {type(m)}")

    #measure distances between all points in M
    distances = []
    for i,point in enumerate(M):
        distances.append([m.distanceTo(M[i]),M[i]])
    distances=sorted(distances,key=lambda x: x[0]) #sort distances numerically
    for i in range(1,5): #ignore the point matching itself
        m.addNeighbour(distances[i][1],distances[i][0]) #add four nearest neighbours to point

def self_test():
    pointList = []
    for i in range(10):
        for j in range(1,10,3):
            pointList.append(Point2D(i,j))
    for i in range(len(pointList)):
        four_nearest_neighbours(pointList,pointList[i])

if __name__ == "__main__":
    self_test()




