try:
    from .Point import Point2D, Cluster, Pair, LocalCluster
except:
    from Point import Point2D,Cluster,Pair,LocalCluster
import numpy as np
import matplotlib.pyplot as plt
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
    for i in range(0,4):
        C.addPoint(distances[i][1]) #add three nearest neighbours to point
    return C

def generateLocalCluster(M,m,k=15,threshold=2.0):
    """ Generate a cluster with k points at coordinate origin point m"""
    if k>len(M):
        k=len(M)-1
    distances=[]
    C = LocalCluster(m,threshold=threshold) #initialize origin
    for i,point in enumerate(M):
        distances.append([m.distanceTo(M[i]),M[i]])
    distances=sorted(distances,key=lambda x:x[0])
    for i in range(1,k+1):
        C.addNeighbour(distances[i][1])
    return C

def pairs_from_vector(V1,V2):
    pairs = []
    for i in range(len(V2)):
        pairs.append(Pair(V1[i],V2[i]))
    return pairs

def evaluate_match(Kf):
    sigma = 2.0
    omit = 0
    E=0
    for i in range(len(Kf)):
        dist = Kf[i].distance()
        if dist>3.0:
            omit+=1
            continue
        else:
            E+=dist
    Ematch = E/(sigma**2.0)+omit
    return Ematch

def generate_key_feature_list(M,D):
    """
    M: List of model points [Point2D]
    D: List of data points [Point2D]
    """
    f=0
    K=[]
    Cm=[]
    Cd=[]
    for i in range(len(M)):
        Cm.append(four_nearest_neighbours(M,M[i]))
        for j in range(len(D)):
            Cd.append(four_nearest_neighbours(D,D[j]))
            P = Cd[-1].getPermutations()
            for k in range(len(P)):
                Kf = [Pair(M[i],D[i])]
                Kf += pairs_from_vector(Cm[i],P[k])
                Ematch = evaluate_match(Kf)
                K.append([Kf,Ematch])
                f+=1
    ret = sorted(K,key=lambda x: x[1])
    return ret

def visualizeLocalCluster(C):
    x= [C[0][0]]
    y= [C[0][1]]
    for j in range(len(C.neighbours)):
        x.append(x[0]+C.neighbours[j][0])
        y.append(y[0]+C.neighbours[j][1])
    plt.scatter(x,y)
    plt.scatter(x[0],y[0])
    plt.show()

def visualizeLocalClusters(C1,C2):
    x1= [C1[0][0]]
    y1= [C1[0][1]]
    x2= [C1[0][0]]
    y2= [C1[0][1]]
    for j in range(len(C1.neighbours)):
        x1.append(x1[0]+C1.neighbours[j][0])
        y1.append(y1[0]+C1.neighbours[j][1])
    for j in range(len(C2.neighbours)):
        x2.append(x2[0]+C2.neighbours[j][0])
        y2.append(y2[0]+C2.neighbours[j][1])
    plt.scatter(x1,y1,label="C1")
    plt.scatter(x2,y2,label="C2")
    plt.scatter(x1[0],y1[0])
    plt.scatter(x2[0],y2[0])
    plt.legend()
    plt.show()

def samSearch(M,D,data_kNN=7,model_kNN=7,match_threshold=1.0,variance_yaw=0.0,offset_yaw=0.0):
    """
    This is currently an exhaustive search
    TODO: Implement a magnitude-based preferential search with an end criterion (we only need 1 star after all)
    """
    model_cluster = None
    data_cluster = None
    Cm=[] #list of local clusters from model
    Cd=[] #list of local clusterd from data
    candidates = []
    for i in range(len(D)):
        Cd.append(generateLocalCluster(D,D[i],k=data_kNN,threshold=match_threshold))
    for j in range(len(M)):
        Cm.append(generateLocalCluster(M,M[j],k=model_kNN,threshold=match_threshold))
    for i in range(len(Cd)): #for each data cluster
        bestTheta = 0
        bestDistance = np.inf
        bestNumberOfMatches = 0
        index = 0
        for j in range(len(Cm)): #search each model cluster
            rotatedCd = Cd[i].rotateLocalCluster(offset_yaw)
            distance,numberOfMatches= rotatedCd.compareDistance(Cm[j])
            if numberOfMatches >= bestNumberOfMatches:
                if distance < bestDistance:
                    bestNumberOfMatches=numberOfMatches
                    bestDistance=distance
                    index = j
                    model_cluster = Cm[j]
                    data_cluster = Cd[i]


        #print(f"Matched point {i} ({Cd[i][0][0]:.2f},{Cd[i][0][1]:.2f}) to {index} ({Cm[index][0][0]:.2f},{Cm[index][0][1]:.2f})")
        #print(f"Best theta: {2*np.pi-bestTheta}, Best Distance: {bestDistance}, Best Number of Matches: {bestNumberOfMatches}")
        candidates.append([Cd[i][0],Cm[index][0],bestNumberOfMatches,bestDistance])

    candidates = sorted(candidates,key=lambda x: (x[2],x[3])) #sort first by number of matches, then by distance
    candidates.reverse()

    #compute angular offset for best candidate
    for theta in np.arange(-variance_yaw, variance_yaw + 0.0002, 0.0002):
        rotatedCd = data_cluster.rotateLocalCluster(offset_yaw + theta)
        distance, numberOfMatches = rotatedCd.compareDistance(model_cluster)
        if distance < bestDistance:
            bestTheta = theta
            bestDistance = distance

    visualizeLocalClusters(data_cluster,data_cluster.rotateLocalCluster(offset_yaw-bestTheta))
    candidates[0].append(bestTheta)

    return candidates
    #visualizeLocalClusters(bestRotatedClust,Cm[index])

def self_test():
    M = []
    D = []
    for i in range(10):
        for j in range(1,10,3):
            M.append(Point2D(i,j))
            D.append(Point2D(i,j).transformPoint(0,2,4.9))
    for point in D:
        print(point[0],point[1])
    samSearch(M,D)
    #generate_key_feature_list(M,D)


if __name__ == "__main__":
    self_test()




