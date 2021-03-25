from Point import Point2D, Cluster
import numpy as np
def four_nearest_neighbours(M,m):
    if not (isinstance(M,list)):
        print(f"WARNING: Expected argument M to be of type 'list', received {type(M)}")
        return
    if not (isinstance(m,Point2D)):
        print(f"WARNING: Expected argument m to be of type 'Point2D', received {type(m)}")

    #measure distances between all points in M
    distances = []
    C = Cluster()
    for i,point in enumerate(M):
        distances.append([m.distanceTo(M[i]),M[i]])
    distances=sorted(distances,key=lambda x: x[0]) #sort distances numerically
    for i in range(1,4): #ignore the point matching itself
        C.addPoint(distances[i][1]) #add three nearest neighbours to point
    return C

def generate_key_feature_list(M,D):
    """
    M: List of model points [Point2D]
    D: List of data points [Point2D]
    """
    f=0
    Cm=[]
    Cd=[]
    for i in range(len(M)):
        percentage = i/float(len(M))
        print(f"{percentage*100:.2f}%")

        Cm.append(four_nearest_neighbours(M,M[i]))
        for j in range(len(D)):
            Cd.append(four_nearest_neighbours(D,D[j]))
            P = Cd[-1].getPermutations()

def self_test():
    M = []
    D = []
    for i in range(10):
        for j in range(1,10,3):
            M.append(Point2D(i,j))
            D.append(Point2D(j,i))
    for i in range(len(M)):
        generate_key_feature_list(M,D)

if __name__ == "__main__":
    self_test()




