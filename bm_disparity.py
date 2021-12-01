import cv2
import numpy as np 
import stereo_setting as stset

import matplotlib
import matplotlib.pyplot as plt

SWS=45
PFS=25
PFC=13
MDS=5
NOD=384
TTH=100
UR=1
SR=32
SPWS=31
BS = 15

################################################################################################

def draw_lines(image, hori):
    y,count = 0,0
    while(count<40):
        image = cv2.line(image, (0,50+y), (hori,50+y), (y,count*4,count*2), 2)
        y+=60
        count+=1

    return image
#enddef

################################################################################################

def filtered_disparity_map(rectL, rectR):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
    #grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    #grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    left_matcher = cv2.StereoBM_create(numDisparities=16, blockSize=BS)
    #sbm.SADWindowSize = SWS
    left_matcher.setPreFilterType(1)
    left_matcher.setPreFilterSize(PFS)
    left_matcher.setPreFilterCap(PFC)
    left_matcher.setMinDisparity(MDS)
    left_matcher.setNumDisparities(NOD)
    #left_matcher.setBlockSize(BS)
    left_matcher.setTextureThreshold(TTH)
    left_matcher.setUniquenessRatio(UR)
    left_matcher.setSpeckleRange(SR)
    left_matcher.setSpeckleWindowSize(SPWS)
    depth = left_matcher.compute(grayL, grayR)

    # Used for the filtered image
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher) # Create another stereo for right this time

    # WLS FILTER Parameters
    lmbda = 80000
    sigma = 1.8
    visual_multiplier = 1.0
    
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    dispL = left_matcher.compute(grayL,grayR)
    dispR = right_matcher.compute(grayR,grayL)
    dispL = np.int16(dispL)
    dispR = np.int16(dispR)

    filteredImg = wls_filter.filter(dispL, grayL, None, dispR)  # important to put "imgL" here!!!
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    filteredImg = np.uint8(filteredImg)
    cv2.imwrite('./filteredImg.png', filteredImg)

    local_max = filteredImg.max()
    local_min = filteredImg.min()
    disparity_grayscale = (filteredImg-local_min)*(65535.0/(local_max-local_min))
    disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
    disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
    return filteredImg
#enddef

################################################################################################

def disparity_map(rectL, rectR):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS
    rectL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    rectR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
    print(rectL.shape[:2])

    sbm = cv2.StereoBM_create(numDisparities=16, blockSize=BS)
    #sbm.SADWindowSize = SWS
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(PFS)
    sbm.setPreFilterCap(PFC)
    sbm.setMinDisparity(MDS)
    sbm.setNumDisparities(NOD)
    #sbm.setBlockSize(BS)
    sbm.setTextureThreshold(TTH)
    sbm.setUniquenessRatio(UR)
    sbm.setSpeckleRange(SR)
    sbm.setSpeckleWindowSize(SPWS)

    disparity = sbm.compute(rectL, rectR)
    norm_coeff = 255 / disparity.max()
    return disparity * norm_coeff / 255
#enddef

def color_disparity_map(rectL, rectR):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS
    rectL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    rectR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    sbm = cv2.StereoBM_create(numDisparities=16, blockSize=BS)
    #sbm.SADWindowSize = SWS
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(PFS)
    sbm.setPreFilterCap(PFC)
    sbm.setMinDisparity(MDS)
    sbm.setNumDisparities(NOD)
    #sbm.setBlockSize(BS)
    sbm.setTextureThreshold(TTH)
    sbm.setUniquenessRatio(UR)
    sbm.setSpeckleRange(SR)
    sbm.setSpeckleWindowSize(SPWS)

    disparity = sbm.compute(rectL, rectR)

    cv2.imwrite('./disparity.png', disparity)

    local_max = disparity.max()
    local_min = disparity.min()
    disparity_grayscale = (disparity-local_min)*(65535.0/(local_max-local_min))
    disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
    disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
    return disparity_color
#enddef

def disp2depth(disparity, cam_param):
    print('cx: ', cam_param[0], '\ncy: ', cam_param[1], '\nf: ', cam_param[2], '\nb: ', cam_param[3])
    
    Z = cam_param[2]*(cam_param[3])/disparity
    plt.imshow(Z)
    plt.title("Depth Map")
    plt.savefig('./Z.png')
    plt.show()
#enddef

def sift_kp_matching(rectL, rectR):
    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(rectL,None)
    kp2, des2 = orb.detectAndCompute(rectR,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    # Draw first 10 matches.
    img3 = cv2.drawMatches(rectL,kp1,rectR,kp2,matches[:20], None, flags=2)
    plt.imshow(img3)
    plt.title("SIFT MATCHING")
    plt.savefig("./sift.png")
    plt.show()
#enddef

################################################################################################

def main():
    pic_name = "9"
    imgL, imgR = cv2.imread("./21_1080p_L_pot/"+pic_name+"_L_.png"), cv2.imread("./21_1080p_R_pot/"+pic_name+"_R_.png")
    #imgL, imgR = cv2.imread("./21_1080p_L_data/0_L_.png"), cv2.imread("./21_1080p_R_data/0_R_.png")
    print(imgL.shape[:2])
    print('IMAGES LOADED')
    print(100*'#')

    #############################################################################################

    vert, hori = imgL.shape[:2]
    left_stereo_map, right_stereo_map, Q = stset.st_maps("./21_1080p_calib_params/", (hori, vert))
    print('MAPS COMPUTED')
    print(100*'#')

    #print(Q)
    cx = -Q[0,3]
    cy = -Q[1,3]
    f = Q[2,3]
    b = 1/Q[3,2]
    cam_param = (cx, cy, f, b)


    #############################################################################################

    rectL, rectR = stset.st_rectify(imgL, imgR, left_stereo_map, right_stereo_map)
    #rectL, rectR = cv2.imread("./rectL.png"), cv2.imread("./rectR.png")
    print('RECTIFIED')
    print(100*'#')

    #rectL, rectR = cv2.imread("./rectifyed_left.jpg", 0), cv2.imread("./rectifyed_right.jpg", 0)

    resize_scale = 3
    scaled = (int(hori/resize_scale), int(vert/resize_scale))
    rectLR = np.concatenate((cv2.resize(draw_lines(rectL.copy(), hori), scaled), cv2.resize(draw_lines(rectR.copy(), hori), scaled)), axis=1)
    plt.imshow(rectLR, 'gray')
    plt.title("Rectified")
    cv2.imwrite("./21_1080p_L_pot_rect/"+pic_name+"_L.png", rectL)
    cv2.imwrite("./21_1080p_R_pot_rect/"+pic_name+"_R.png", rectR)
    #cv2.imshow('rectL', cv2.resize(draw_lines(rectL.copy(), hori), scaled))
    #cv2.imshow('rectR', cv2.resize(draw_lines(rectR.copy(), hori), scaled))

    #############################################################################################
    
    disparity = disparity_map(rectL, rectR)
    fil_disparity = filtered_disparity_map(rectL, rectR)
    col_disparity = color_disparity_map(rectL, rectR)
    #cv2.imshow('disparity', cv2.resize(disparity, scaled))
    #cv2.imwrite('./disparity.png', disparity)
    
    imgs = []
    imgs.append(rectL)
    imgs.append(disparity)
    imgs.append(np.fliplr(fil_disparity.reshape(-1,3)).reshape(fil_disparity.shape))
    imgs.append(np.fliplr(col_disparity.reshape(-1,3)).reshape(col_disparity.shape))

    row,col = 2,2
    f, axarr = plt.subplots(row,col,figsize=(10,10))
    f.tight_layout()
    plt.title("Disparity Maps")

    axarr[0,0].imshow(imgs[0], 'gray'),axarr[0,0].set_title('rectL', color='r', size=15), axarr[0,0].axis('off')
    axarr[0,1].imshow(imgs[1], 'gray'),axarr[0,1].set_title('disparity', color='r', size=15), axarr[0,1].axis('off')

    axarr[1,0].imshow(imgs[2]),axarr[1,0].set_title('fil_disparity', color='r', size=15), axarr[1,0].axis('off')
    axarr[1,1].imshow(imgs[3]),axarr[1,1].set_title('col_disparity', color='r', size=15), axarr[1,1].axis('off')

    plt.show()

    print('DISP MAPS COMPUTED')
    print(100*'#')

    #############################################################################################

    disp2depth(fil_disparity, cam_param)
    sift_kp_matching(rectL, rectR)

    print('DEPTH AND STEREO MATCHING DONE')
    print(100*'#')

    #############################################################################################

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#enddef

################################################################################################

if __name__ == "__main__":
    main()


