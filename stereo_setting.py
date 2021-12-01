import numpy as np
import cv2

def st_maps(calib_params, im_dim):
    MLS = np.load(calib_params+'MLS.npy')
    MRS = np.load(calib_params+'MRS.npy')
    dLS = np.load(calib_params+'dLS.npy')
    dRS = np.load(calib_params+'dRS.npy')
    RL = np.load(calib_params+'RL.npy')
    PL = np.load(calib_params+'PL.npy')
    RR = np.load(calib_params+'RR.npy')
    PR = np.load(calib_params+'PR.npy')
    Q = np.load(calib_params+'Q.npy')
    
    left_stereo_map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL, im_dim, cv2.CV_16SC2) 
    right_stereo_map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, im_dim, cv2.CV_16SC2)

    return left_stereo_map, right_stereo_map, Q

def st_rectify(imgL, imgR, left_stereo_map, right_stereo_map):
    rectL = cv2.remap(imgL,left_stereo_map[0], left_stereo_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  
    rectR = cv2.remap(imgR,right_stereo_map[0], right_stereo_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    return rectL, rectR