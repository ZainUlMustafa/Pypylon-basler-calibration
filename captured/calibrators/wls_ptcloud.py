# Followed 
# https://medium.com/@omar.ps16/stereo-3d-reconstruction-with-opencv-using-an-iphone-camera-part-iii-95460d3eddf0
import stereo_setting as stset
import cv2
import numpy as np

BS=9
MDS=0
NOD=432
UR=1
SPWS=128 
SR=10
DMD=5
P1 = 8*3*BS**2
P2 = 32*3*BS**2 

# Pothole Params (DONT TOUCH)
'''
BS=15#17 
MDS=0
NOD=368-MDS
UR=5 
SPWS=230 
SR=23
DMD=5 
P1 = 8*3*BS**2
P2 = 32*3*BS**2 
'''

def disparity_map(left_image, right_image):
    grayL = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    global BS, MDS, NOD, UR, SPWS, SR, DMD, P1, P2
    window_size = BS
    min_disp = MDS
    num_disp = NOD
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        #SADWindowSize=window_size,
        uniquenessRatio=UR,#10,
        speckleWindowSize=SPWS,#100,
        speckleRange=SR,#32,
        disp12MaxDiff=DMD,#1,
        P1=8*3*window_size**2,
        P2=32*3*window_size**2,
        #fullDP=False
    )
    return stereo.compute(grayL, grayR).astype(np.float32) / 16.0
#enddef

def filtered_disparity_map(rectL, rectR):
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
    global BS, MDS, NOD, UR, SPWS, SR, DMD, P1, P2
    #grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    #grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    # Create StereoSGBM and prepare all parameters
    '''
    window_size = 7
    min_disp = 1
    num_disp = (16*35)+1-min_disp
    left_matcher = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = window_size,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32,
        disp12MaxDiff = 5,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2)
    '''
    # left_matcher = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)
    # left_matcher.setBlockSize(BS)
    # left_matcher.setMinDisparity(MDS)
    # left_matcher.setNumDisparities(NOD)
    # left_matcher.setUniquenessRatio(UR)
    # left_matcher.setSpeckleWindowSize(SPWS)
    # left_matcher.setSpeckleRange(SR)
    # left_matcher.setDisp12MaxDiff(DMD)
    # left_matcher.setP1(P1)
    # left_matcher.setP2(P2)
    window_size = 5
    min_disp = 16*21
    num_disp = (16*1)
    
    left_matcher = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = window_size,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32,
        disp12MaxDiff = 5,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2)
        
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
    cv2.imwrite('./depth.png', depth)

    local_max = filteredImg.max()
    local_min = filteredImg.min()
    disparity_grayscale = (filteredImg-local_min)*(65535.0/(local_max-local_min))
    disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
    disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
    return filteredImg
#enddef

def disparity_map(rectL, rectR):
    global BS, MDS, NOD, UR, SPWS, SR, DMD, P1, P2
    rectL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    rectR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
    print(rectL.shape[:2])
    '''
    stereoMatcher = cv2.StereoSGBM_create()
    stereoMatcher.setMinDisparity(0)
    stereoMatcher.setNumDisparities(16*10)
    stereoMatcher.setBlockSize(5)
    stereoMatcher.setSpeckleRange(32)
    stereoMatcher.setSpeckleWindowSize(100)
    depth = stereoMatcher.compute(rectL, rectR)
    '''
    sbm = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)
    sbm.setBlockSize(BS)
    sbm.setMinDisparity(MDS)
    sbm.setNumDisparities(NOD)
    sbm.setUniquenessRatio(UR)
    sbm.setSpeckleWindowSize(SPWS)
    sbm.setSpeckleRange(SR)
    sbm.setDisp12MaxDiff(DMD)
    sbm.setP1(P1)
    sbm.setP2(P2)

    disparity = sbm.compute(rectL, rectR)
    disparity = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    disparity = np.uint8(disparity)
    return disparity
#enddef

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    print("verts => ", len(verts))
    verts, colors = filter_infinity(verts, colors)
    verts = np.hstack([verts, colors])
    print("verts stack => ", verts.shape)
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
#enddef

####################################################################################
####################################################################################

def gen_3D_point_cloud(Q, h, w, disparity, img_left, img_right):
    cx = -Q[0,3]
    cy = -Q[1,3]
    focus_px = Q[2,3] #focus_px = physical focal length (mm) * size of imager element (px/mm)
    tx = 1/Q[3,2]
    print(focus_px, tx)

    Q1 = np.array([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, focus_px],
        [0, 0, -1/tx, 0]
    ])

    #This transformation matrix is derived from Prof. Didier Stricker's power point presentation on computer vision. 
    #Link : https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws14-15/3DCV_lec01_camera.pdf
    Q2 = np.float32([
        [1, 0,  0,                  0],
        [0, -1, 0,                  0],
        [0, 0,  focus_px*0.0078,    0], #Focal length multiplication obtained experimentally. 
        [0, 0,  0,                  1]
    ])
    
    #This Transformation keeps baseline and optical centers into consideration
    Q3 = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -focus_px], # so that y-axis looks up
                    [0, 0, 1/tx,      0]])

    Q4 = np.float32([
        [1, 0, 0, -cx],
        [0, -1, 0, cy],
        [0, 0, 0, -focus_px],
        [0, 0, 1/tx, 0]
    ])

    points = cv2.reprojectImageTo3D(disparity, Q4)
    #cv2.imshow("d", points)
    #cv2.waitKey(0)
    colors = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)

    mask = disparity > disparity.min()
    out_points = points[mask]
    out_colors = colors[mask]
    print("len => ", len(out_points))

    '''cv2.imshow('left', img_left)
    cv2.imshow('right', img_right)
    cv2.waitKey(0)'''
    write_ply('out_new.ply', out_points, out_colors)
    print('%s saved' % 'out_new.ply')
#enddef

def filter_infinity(coordinates, colors):
    """Filter infinite distances from ``PointCloud.``"""
    mask = coordinates[:, 2] > coordinates[:, 2].min()
    coords = coordinates[mask]
    colors = colors[mask]
    return coords, colors

####################################################################################
####################################################################################


def main():
    imgL, imgR = cv2.imread("../test/left/900_L_.png"), cv2.imread("../test/right/900_R_.png")
    print(imgL.shape[:2])
    print('IMAGES LOADED')
    print(100*'#')

    #############################################################################################

    vert, hori = imgL.shape[:2]
    left_stereo_map, right_stereo_map, Q = stset.st_maps("./calibParams/", (hori, vert))
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
    
    disp = filtered_disparity_map(rectL, rectR)
    #cv2.imshow("disp", cv2.resize(disp, scaled))
    #cv2.waitKey(0)

    gen_3D_point_cloud(Q, vert, hori, disp, rectL, rectR)
    print("DONE")
#enddef

if __name__ == "__main__":
    main()
#endif