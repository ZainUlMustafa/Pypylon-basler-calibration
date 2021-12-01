# Followed 
# https://medium.com/@omar.ps16/stereo-3d-reconstruction-with-opencv-using-an-iphone-camera-part-iii-95460d3eddf0
import stereo_setting as stset
import cv2
import numpy as np

SWS=5 
PFS=5 
PFC=29 
MDS=-25 
NOD=128 
TTH=100
UR=10 
SR=15 
SPWS=100
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
    #verts, colors = filter_infinity(verts, colors)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
#enddef

####################################################################################
####################################################################################

def gen_3D_point_cloud(Q, h, w, disparity, img_left, img_right):
    cx = -Q[0,3]
    cy = -Q[1,3]
    focus_mm = Q[2,3]
    tx = 1/Q[3,2]

    Q1 = np.array([
        [1, 0, 0, -w/2],
        [0, 1, 0, -h/2],
        [0, 0, 0, focus_mm],
        [0, 0, -1/tx, 0]
    ])

    #This transformation matrix is derived from Prof. Didier Stricker's power point presentation on computer vision. 
    #Link : https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws14-15/3DCV_lec01_camera.pdf
    Q2 = np.float32([
        [1,0,0,0],
        [0,-1,0,0],
        [0,0,focus_mm*0.001,0], #Focal length multiplication obtained experimentally. 
        [0,0,0,1]
    ])
    
    #This Transformation keeps baseline and optical centers into consideration
    Q3 = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -focus_mm], # so that y-axis looks up
                    [0, 0, 1/tx,      0]])

    points = cv2.reprojectImageTo3D(disparity, Q2)
    #cv2.imshow("d", points)
    #cv2.waitKey(0)
    colors = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)

    mask = disparity > disparity.min()
    out_points = points[mask]
    out_colors = colors[mask]

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
    img_left = cv2.imread('./tsukubaL.bmp')
    img_right = cv2.imread('./tsukubaR.bmp')

    vert, hori = img_left.shape[:2]
    _, _, Q = stset.st_maps("./21_1080p_calib_params/", (hori, vert))

    disp = disparity_map(img_left, img_right)
    cv2.imshow("disp", disp)
    cv2.waitKey(0)

    gen_3D_point_cloud(Q, vert, hori, disp, img_left, img_right)
    print("DONE")
#enddef

if __name__ == "__main__":
    main()
#endif