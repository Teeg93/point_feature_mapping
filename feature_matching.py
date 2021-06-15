import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from datetime import datetime
try:
    from .functions import *
except:
    from functions import *
import time

def threshold(grayscale_img,threshold=100,num_stars=25,tolerance=3,depth=0):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    ret,thresholded_image = cv2.threshold(grayscale_img,threshold,255,cv2.THRESH_BINARY)
    res = cv2.morphologyEx(thresholded_image,cv2.MORPH_OPEN,kernel)
    cnts, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    intensitySortingList=[]
    for contour in cnts:
        mask = np.zeros(thresholded_image.shape,np.uint8)
        cv2.drawContours(mask,contour,-1,255,-1)
        mean = cv2.mean(grayscale_img,mask=mask)
        intensitySortingList.append([contour,mean])
    sortedContours=sorted(intensitySortingList,key=lambda x: x[1]) #sort by average intensity
    sortedContours.reverse()
    if num_stars > len(sortedContours):
        num_stars = len(sortedContours)
    returnList = []
    for i in range(num_stars):
        returnList.append(sortedContours[i][0])
    return returnList

class ImageMeta:
    def __init__(self,width,height,focal_length,pixel_size,hfov=None,vfov=None):
        self.width = width
        self.height = height
        self.focal_length = focal_length
        self.pixel_size = pixel_size
        if not hfov:
            self.hfov = np.degrees(self.compute_fov(width,pixel_size,focal_length))
            #print(f"Horizontal FOV: {self.hfov}")
        else:
            self.hfov = hfov
        if not vfov:
            self.vfov = np.degrees(self.compute_fov(height,pixel_size,focal_length))
            #print(f"Vertical FOV:   {self.vfov}")
        else:
            self.vfov = vfov
        self.x_mid = self.width / 2
        self.y_mid = self.height / 2

    def compute_fov(self,pixels,pixel_size,focal_length):
        x = (pixels/2.0)*pixel_size
        fov = 2*np.arctan(x/focal_length)
        return fov

    def point_to_angle(self,p):
        dx = p[0] - self.x_mid
        dy = p[1] - self.y_mid
        theta_x = np.degrees(np.arctan(dx*self.pixel_size/self.focal_length))
        theta_y = np.degrees(np.arctan(dy*self.pixel_size/self.focal_length))
        return(theta_x,theta_y)

    def angle_to_point(self,angle):
        dx = (self.focal_length*np.tan(np.radians(angle[0])))/self.pixel_size
        dy = (self.focal_length*np.tan(np.radians(angle[1])))/self.pixel_size
        x = int(self.x_mid+dx)
        y = int(self.y_mid+dy)
        return(x,y)

    def angle_between_points(self,p1,p2):
        p1 = self.point_to_angle(p1)
        p2 = self.point_to_angle(p2)
        theta_x = p2[0]-p1[0]
        theta_y = p2[1]-p1[1]
        return np.sqrt(theta_x**2.0+theta_y**2.0)

    def distance_and_angle_between_points(self,p1,p2):
        p1 = self.point_to_angle(p1)
        p2 = self.point_to_angle(p2)
        theta_x = p2[0]-p1[0]
        theta_y = p2[1]-p1[1]
        distance = np.sqrt(theta_x**2.0+theta_y**2.0)
        arg = np.arctan2(theta_y,theta_x)
        return (distance,arg)

    def cartesian_angle_between_points(self,p1,p2):
        p1 = self.point_to_angle(p1)
        p2 = self.point_to_angle(p2)
        theta_x = p2[0]-p1[0]
        theta_y = p2[1]-p1[1]
        return(theta_x,theta_y)

    def get_image_centre(self):
        return (self.x_mid,self.y_mid)

def compare_arrays(arr1,arr2):
    score = 0
    for i in range(len(arr1)):
        for j in range(len(arr2)):
            diff = (arr1[i]-arr2[j])
            if diff == 0:
                continue
            score += 1/((arr1[i]-arr2[j])**2.0)
    return score


def compute_distance_cartesian(p1,p2):
    D = np.sqrt((p2[0]-p1[0])**2.0+(p2[1]-p1[1])**2.0)
    return D

def compute_distance_polar(p1,p2):
    D = np.sqrt(p1[0]**2.0+p2[0]**2.0 - 2*p1[0]*p2[0]*np.cos(p1[1]-p2[1]))
    return D

def compute_distance_score(arr1,arr2,threshold=np.inf):
    score = 0
    for i in range(len(arr1)):
        min_dist = np.inf
        for j in range(len(arr2)):
            D = compute_distance_polar(arr1[i],arr2[j])
            if D < min_dist:
                min_dist = D
        if D < threshold: #only count if below a given threshold
            score+=1/(min_dist**2.0)
    return score


def computeAngularOffset(im1,im2,width=3280,height=2464,focal_length=3.04e-3,pixel_size=1.12e-6,display=False,data_threshold=60,model_threshold=50,star_match_threshold=1.0,variance_yaw=0.0,output_path=""):
    """
    :param im1: image containing the data point
    :param im2: image of the model
    :param width: image width
    :param height: image height
    :param focal_length: focal length of camera (m)
    :param pixel_size: pixel size of camera (m)
    :param display: true:display results, false: do not display result
    :param data_threshold: image binary threshold value for data
    :param model_threshold: image binary threshold value for model
    :param star_match_threshold: tolerable angular displacement between constellations for a match
    :param variance_yaw: variance in yaw
    :return: theta_x, theta_y
    """
    #Threshold the images
    imageMeta = ImageMeta(width,height,focal_length=focal_length,pixel_size=pixel_size)
    data_cnts = threshold(im1,threshold=data_threshold,num_stars=30)
    model_cnts = threshold(im2,threshold=model_threshold,num_stars=60)

    data_stars = []
    model_stars = []
    i=0
    for c in data_cnts:
        if len(c) > 2: #if more than 2 pixels in size
            M = cv2.moments(c)
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            except:
                continue
        else:
            cX = c[0][0][0] #use first pixel value as origin
            cY = c[0][0][1]
        data_stars.append([cX,cY])
        i += 1
    i = 0
    for c in model_cnts:
        if len(c)>2:
            M = cv2.moments(c)
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            except:
                continue
        else:
            cX = c[0][0][0]
            cY = c[0][0][1]
        model_stars.append([cX,cY])
        i+=1

    M = [] #list of model points
    D = [] #list of data points
    for i in range(len(data_stars)):
        p = imageMeta.point_to_angle(data_stars[i])
        D.append(Point2D(p[0],p[1]))

    for i in range(len(model_stars)):
        p = imageMeta.point_to_angle(model_stars[i])
        M.append(Point2D(p[0],p[1]))

    im1=cv2.cvtColor(im1,cv2.COLOR_GRAY2BGR)
    im2=cv2.cvtColor(im2,cv2.COLOR_GRAY2BGR)
    data_kNN = 15 #how many neighbours per cluser
    model_kNN = 100

    now = time.time()
    #run the matching operation
    candidates = samSearch(M,D,data_kNN=data_kNN,model_kNN=model_kNN,match_threshold=star_match_threshold,variance_yaw=variance_yaw)
    search_time = time.time()-now
    #print(f"Search time: {search_time}")

    bestNumMatches = candidates[0][2]
    disp_x = []
    disp_y = []
    point_0 = (candidates[0][0][0],candidates[0][0][1])
    point_1 = (candidates[0][1][0],candidates[0][1][1])
    for i,candidate in enumerate(candidates):
        if (candidate[2] < bestNumMatches and i > 4): #only use candidates with high number of matches
            break
        disp_x.append(candidate[1][0]-candidate[0][0])
        disp_y.append(candidate[1][1]-candidate[0][1])
        if i==0:
            color=(255,0,0)
        else:
            color=(0,255,0)
        point0 = (imageMeta.angle_to_point([candidate[0][0],candidate[0][1]]))
        point1 = (imageMeta.angle_to_point([candidate[1][0],candidate[1][1]]))
        #cv2.line(im2,point0,point1,color,1)
        #cv2.putText(im1,f"{i}",(point0[0]-10,point0[1]+90),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,255),2)
        cv2.circle(im1,point0,10,color,2)
        #cv2.putText(im2,f"{i}",(point1[0]-10,point1[1]+90),cv2.FONT_HERSHEY_SIMPLEX,3,(255,255,255),2)
        cv2.circle(im2,point1,10,color,2)
        #cv2.circle(im2,point0, 10, color, 2)
    mean_x = np.mean(disp_x)
    mean_y = np.mean(disp_y)
    #print(f"Angular displacement: {mean_x:.2f},{mean_y:.2f}")

    if display:
        f=plt.figure()
        f.add_subplot(1,2,1)
        plt.imshow(im1)
        plt.title("Data")
        f.add_subplot(1,2,2)
        plt.imshow(im2)
        plt.title("Model")
        plt.show()
    if output_path:
        date = datetime. now().strftime("%Y_%m_%d-%H:%M:%S")
        data_path=os.path.join(output_path,date+"_data.jpg")
        model_path=os.path.join(output_path,date+"_model.jpg")
        cv2.imwrite(data_path,im1)
        cv2.imwrite(model_path,im2)
    return point_0,point_1

if __name__ == "__main__":
    im1 = cv2.imread('real.jpeg',cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread('sim.png',cv2.IMREAD_GRAYSCALE)
    computeAngularOffset(im1,im2,star_match_threshold=0.3,variance_yaw=0,display=True)
