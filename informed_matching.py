# -*- coding: utf-8 -*-
"""
STRADA
INFORMED KPT MATCHING ROUTINE FOR POTHOLE IMAGES

Moeez.Strada
"""
##############################################################################

### INCLUDING HEADERS ###

import cv2
import numpy as np
import matplotlib.pyplot as plt
import stereo_setting as stset
print(cv2.__version__)

##############################################################################

### FUNCTIONS ###

#kpt matching
def kpt_matching(img_l, img_r, bf, n_matches):
    
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img_l, None)
    kp2, des2 = orb.detectAndCompute(img_r, None)
    
    # Proceed if kpts are detected
    if (len(kp1) > 0) & (len(kp2) > 0):
        
        # Match descriptors
        matches = bf.match(des1,des2)
        
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        
        # Extract kpt locations from matched pts
        list_kp1 = np.asarray([kp1[mat.queryIdx].pt for mat in matches[:n_matches]])
        list_kp2 = np.asarray([kp2[mat.trainIdx].pt for mat in matches[:n_matches]])
        
    else:
        list_kp1 = []
        list_kp2 = []
        
    # Convert data type
    list_kp1 = np.array(list_kp1)
    list_kp2 = np.array(list_kp2)

    return list_kp1, list_kp2

##############################################################################

### DECLARE VARIABLES ###

# variables
grid_sz = 100
n_matches = 500  
list_kp1 = []
list_kp2 = []
dense_kp1 = None
dense_kp2 = None

# Initiate SIFT detector
orb = cv2.ORB_create()

# create Brute-Force Matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

##############################################################################

### LOAD IMAGES ###

# Rectifying the images
imgL, imgR = cv2.imread("./singlerun_2020/20_R_.png", 0), cv2.imread("./singlerun_2020/20_L_.png", 0)
print(imgL.shape[:2])
print('IMAGES LOADED')
print(100*'#')

vert, hori = imgL.shape[:2]
left_stereo_map, right_stereo_map, _ = stset.st_maps("./basler_calib_params_2020/", (hori, vert))
print('MAPS COMPUTED')
print(100*'#')

rectL, rectR = stset.st_rectify(imgL, imgR, left_stereo_map, right_stereo_map)
print('RECTIFIED')
print(100*'#')

img_l, img_r = rectL.copy(), rectR.copy() 

##############################################################################
# First Step in Paper
### PRELIMINARY MATCHING ###

list_kp1, list_kp2 = kpt_matching(img_l, img_r, bf, n_matches)

##############################################################################

### TRANSFORMATION bw IMAGES ###

# Finding the mean x & y axes pixel shift bw kpts
delta = list_kp1 - list_kp2
x_delta = int(np.floor(np.abs(np.mean(delta[:,0]))))
y_delta = int(np.floor(np.abs(np.mean(delta[:,1]))))

# Zero-value check for x_delta
if x_delta == 0:
    x_delta += 1
    
##############################################################################

### DEFINE RANGE FOR ROI ###
# Second Step in Paper    
# Define the pixel range for x-axis ROI
# range(start, stop, step)

if x_delta>=0:
    x_range = range(0, img_r.shape[0] - x_delta - grid_sz, grid_sz)
elif x_delta<0:
    x_range = range(np.abs(x_delta), img_r.shape[0] - grid_sz, grid_sz)

# Define the pixel range for y-axis ROI
if y_delta>=0:
    y_range = range(0, img_r.shape[1] - y_delta - grid_sz, grid_sz)
elif y_delta<0:
    y_range = range(np.abs(y_delta), img_r.shape[1] - grid_sz, grid_sz)
elif y_delta == 0:
    y_check = 'True'    
    
##############################################################################

### INFORMED MATCHING ###

dense_kp1 = list_kp1
dense_kp2 = list_kp2

for i in x_range:
    for j in y_range:
        
        ROI_l = img_l[j:j+grid_sz, i:i+grid_sz]
        ROI_r = img_r[j+y_delta:j+grid_sz+y_delta, i+x_delta:i+grid_sz+x_delta]
        
        ROI_kp1, ROI_kp2 = kpt_matching(ROI_l, ROI_r, bf, n_matches)
        
        if (len(ROI_kp1)>0) & (len(ROI_kp1)> 0):
            ROI_kp1 = ROI_kp1 + [(i,j)]
            ROI_kp2 = ROI_kp2 + [(i+x_delta, j+y_delta)]
        
            dense_kp1 = np.vstack((dense_kp1, ROI_kp1))
            dense_kp2 = np.vstack((dense_kp2, ROI_kp2))
        
        print(i,j)


print(len(dense_kp1))
#############################################################################

### DISPLAY RESULTS ###

# Open a new figure
win = plt.figure()

# Add a subplot for left image
win.add_subplot(1,2,1)
implot = plt.imshow(img_l, 'gray')
plt.scatter(dense_kp1[:,0], dense_kp1[:,1], c='g', s=5)
plt.axis('off')

# Add a subplot for right image
win.add_subplot(1,2,2)
implot = plt.imshow(img_r, 'gray')
plt.scatter(dense_kp2[:,0], dense_kp2[:,1], c='r', s=5)
plt.axis('off')

# Display the image
plt.imshow
plt.show()
#plt.waitforbuttonpress(0)
#plt.close('all')
        

#############################################################################

### End of Script
        
##############################################################################        
        
### APPENDIX ###

# Create vector of ones  for homogemous coordinates
#h_v = np.ones(len(list_kp1))
#h_v = np.transpose(h_v)

# Convert to homogenous coordinates
#list_kp1 = np.column_stack((list_kp1, h_v))
#list_kp2 = np.column_stack((list_kp2, h_v))



        # Draw n-strongest matches and display
#        img_out = cv2.drawMatches(img_l,kp1,img_r,kp2,matches[:n_matches], None)
#        cv2.namedWindow('Matches', cv2.WINDOW_NORMAL)
#        cv2.imshow('Matches', img_out)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()   