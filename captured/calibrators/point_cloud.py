from this import d
import cv2
import numpy as np
import matplotlib.pyplot as plt
import struct
#=========================================================
# Stereo 3D reconstruction 
#=========================================================
#Load camera parameters

def write_pointcloud(xyz_points,rgb_points,filename):

    """ creates a .pkl file of the point clouds generated
    """

    assert xyz_points.shape[1] == 3,'Input XYZ points should be Nx3 float array'
    if rgb_points is None:
        rgb_points = np.ones(xyz_points.shape).astype(np.uint8)*255
    assert xyz_points.shape == rgb_points.shape,'Input RGB colors should be Nx3 float array and have same size as input XYZ points'

    # Write header of .ply file
    fid = open(filename,'wb')
    fid.write(bytes('ply\n', 'utf-8'))
    fid.write(bytes('format binary_little_endian 1.0\n', 'utf-8'))
    fid.write(bytes('element vertex %d\n'%xyz_points.shape[0], 'utf-8'))
    fid.write(bytes('property float x\n', 'utf-8'))
    fid.write(bytes('property float y\n', 'utf-8'))
    fid.write(bytes('property float z\n', 'utf-8'))
    fid.write(bytes('property uchar red\n', 'utf-8'))
    fid.write(bytes('property uchar green\n', 'utf-8'))
    fid.write(bytes('property uchar blue\n', 'utf-8'))
    fid.write(bytes('end_header\n', 'utf-8'))

    # Write 3D points to .ply file
    for i in range(xyz_points.shape[0]):
        fid.write(bytearray(struct.pack("fffccc",xyz_points[i,0],xyz_points[i,1],xyz_points[i,2],
                                        rgb_points[i,0].tobytes(),rgb_points[i,1].tobytes(),
                                        rgb_points[i,2].tobytes())))
    fid.close()

# K = np.array([[1733.74 ,0 ,792.27],[ 0, 1733.74 ,541.89],[0, 0, 1]],dtype=np.float32)
calib_params = "./calibParams/"
MLS = np.load(calib_params+'MLS.npy')
MRS = np.load(calib_params+'MRS.npy')
dLS = np.load(calib_params+'dLS.npy')
dRS = np.load(calib_params+'dRS.npy')
R = np.load(calib_params+'R.npy')
T = np.load(calib_params+'T.npy')
RL = np.load(calib_params+'RL.npy')
PL = np.load(calib_params+'PL.npy')

RR = np.load(calib_params+'RR.npy')
PR = np.load(calib_params+'PR.npy')
Q = np.load(calib_params+'Q.npy')
dist = np.zeros((5,1),dtype=np.float32)

cameraMatrix1 = np.array([[886.2360,0,823.1205],[0,888.4556,603.5689],[0,0,1]])
cameraMatrix2 = np.array([[957.3603,0,842.2650],[0,962.2251,598.1393],[0,0,1]]) 
distCoeffs1 = np.array([-0.2567 ,0.0652,0,0]) 
distCoeffs2 = np.array([-0.2712 ,0.0607,0,0])

scameraMatrix1 = np.array([[1014.1,0,850.80],[0,1010.300,581.3],[0,0,1]])
scameraMatrix2 = np.array([[888.6949,0,838.8008],[0,880.8479,638.3321],[0,0,1]]) 
sdistCoeffs1 = np.array([-0.0229 ,0.0218,0,0]) 
sdistCoeffs2 = np.array([-0.0350 ,0.0235,0,0])
Rotation = np.array([[0.9998,-0.0097,-0.0190],[0.0077,0.9946,-0.1037],[0.0199,0.1036,0.9944]])
Translation = np.array([354.5368,-77.7998,-179.6456])

#Specify image paths
ldir = "../test/left/"
rdir = "../test/right/"
img_no = 110
img_path1 = ldir+f'{str(img_no)}_L_.png'
img_path2 = rdir+f'{str(img_no)}_R_.png'

#Load pictures
img_r = cv2.imread(img_path1,cv2.IMREAD_GRAYSCALE)
img_l = cv2.imread(img_path2,cv2.IMREAD_GRAYSCALE)
# img_l = cv2.flip(img_l,0)
# img_r = cv2.flip(img_r,0)
plt.subplot(321)
plt.imshow(img_l,"gray")

plt.subplot(322)
plt.imshow(img_r,"gray")

#Get height and width. Note: It assumes that both pictures are the same size. They HAVE to be same size 
h,w = img_r.shape[:2]

# new_camera_matrix1, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix1,distCoeffs1,(w,h),0,(w,h))
# new_camera_matrix2, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix2,distCoeffs2,(w,h),0,(w,h))


# # # #Undistort images
# rectL = cv2.undistort(img_l, MLS,dLS, None, new_camera_matrix1)
# rectR = cv2.undistort(img_r, MRS,dRS, None, new_camera_matrix2)

R1,R2,P1,P2,Q,roi1,roi2 = cv2.stereoRectify(MLS,dLS,MRS,dRS,(w,h),R,T)
left_stereo_map = cv2.initUndistortRectifyMap(MLS,dLS, R1, P1, (w,h), cv2.CV_16SC2) 
right_stereo_map = cv2.initUndistortRectifyMap(MRS,dRS, R2, P2, (w,h), cv2.CV_16SC2)
prectL = cv2.remap(img_l,left_stereo_map[0], left_stereo_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)  
prectR = cv2.remap(img_r,right_stereo_map[0], right_stereo_map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)   
#Stereo Uncalibrated rectification using keypoints
plt.subplot(323)
plt.imshow(prectL,"gray")
plt.subplot(324)

plt.imshow(prectR,"gray")


# sift = cv2.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(prectL, None)
# kp2, des2 = sift.detectAndCompute(prectR, None)
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# search_params = dict(checks=50)   # or pass empty dictionary
# flann = cv2.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(des1, des2, k=2)

# # Keep good matches: calculate distinctive image features
# # Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
# # https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
# matchesMask = [[0, 0] for i in range(len(matches))]
# good = []
# pts1 = []
# pts2 = []

# for i, (m, n) in enumerate(matches):
#     if m.distance < 0.7*n.distance:
#         # Keep this keypoint pair
#         matchesMask[i] = [1, 0]
#         good.append(m)
#         pts2.append(kp2[m.trainIdx].pt)
#         pts1.append(kp1[m.queryIdx].pt)


# pts1 = np.int32(pts1)
# pts2 = np.int32(pts2)


# fundamental_matrix, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

# # We select only inlier points
# pts1 = pts1[inliers.ravel() == 1]
# pts2 = pts2[inliers.ravel() == 1]
# h1, w1 = prectL.shape
# h2, w2 = prectR.shape
# _, H1, H2 = cv2.stereoRectifyUncalibrated(
#     np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
# )

# rectL = cv2.warpPerspective(prectL, H1, (w1, h1))
# rectR = cv2.warpPerspective(prectR, H2, (w2, h2))



# plt.subplot(325)

# plt.imshow(rectL,"gray")

# plt.subplot(326)

# plt.imshow(rectR,"gray")
# plt.show()
# def callback():
# pass
# c2.namedWindow("Disparity Tune",cv2.WINDOW_AUTOSIZE)
# c2.createTrackbar('SGBM_Win_Size','Disparity Tune',5,250,callback)
# c2.createTrackbar('BM_Win_Size','Disparity Tune',5,250,callback)
# c2.createTrackbar('SGBM_Min_Disparity','Disparity Tune',16,1200,callback)
# c2.createTrackbar('BM_Min_Disparity','Disparity Tune',16,1200,callback)
# c2.createTrackbar('BlockSize','Disparity Tune',5,255,callback)
# wile  True:
#    sgbmwin = cv2.getTrackbarPos('SGBM_Win_Size','Disparity Tune')+5
#    if sgbmwin%2==0:
#        sgbmwin=sgbmwin-1
#    bmwin = cv2.getTrackbarPos('BM_Win_Size','Disparity Tune')+5
#    if bmwin%2==0:
#        bmwin=bmwin-1
#    sgbmdisp = cv2.getTrackbarPos('SGBM_Min_Disparity','Disparity Tune')
#    if sgbmdisp%16!=0:
#        sgbmdisp=16+int(sgbmdisp/16)*16
#    bmdisp = cv2.getTrackbarPos('BM_Min_Disparity','Disparity Tune')
#    if bmdisp%16!=0:
#        bmdisp=16+int(bmdisp/16)*16
win_size = 6
min_disp = 0
max_disp = 10 #min_disp * 9
num_disp = max_disp - min_disp # Needs to be divisible by 16
#Create Block matching object. 
stereo = cv2.StereoSGBM_create(
    minDisparity= 5,
    numDisparities = 336,
    blockSize = 3,
    uniquenessRatio = 2,
    speckleWindowSize = 128,
    speckleRange = 10,
    disp12MaxDiff = 5,
    P1 = 216,
    P2 = 864
) 
#Compute disparity map
print ("\nComputing the disparity  map...")

# stereobm = cv2.StereoBM_create(bmdisp,bmwin)
# disparity_bm = stereobm.compute(prectL, prectR)
disparity_sgbm = stereo.compute(prectL, prectR)
#Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
# total = np.concatenate([disparity_bm,disparity_sgbm],1)
# total = cv2.resize (total,(1024,512))
# colormap = plt.get_cmap('plasma')
# heatmap = (colormap(total) * 2**16).astype(np.uint16)[:,:,:3]
# heatmap=cv2.cvtColor(heatmap,cv2.COLOR_RGB2BGR)
# cv2.imshow("Disparity Tune",heatmap)
# k = cv2.waitKey(1) & 0xFF
# if k == 27:
#     break
# #Generate  point cloud. 
# print ("\nGenerating the 3D map...")
# #Get new downsampled width and height 
# # h,w = rectR.shape[:2]
# #Load focal length. 
focal_length = 4
# #Perspective transformation matrix
# #This transformation matrix is from the openCV documentation, didn't seem to work for me. 
# Q = np.float32([[1,0,0,-w/2.0],
#     [0,-1,0,h/2.0],
#     [0,0,0,-focal_length],
#     [0,0,1,0]])
# #This transformation matrix is derived from Prof. Didier Stricker's power point presentation on computer vision. 
# #Link : https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws14-15/3DCV_lec01_camera.pdf
Q2 = np.float32([[1,0,0,0],
    [0,-1,0,0],
    [0,0,focal_length*0.05,0], #Focal length multiplication obtained experimentally. 
    [0,0,0,1]])
# # Third transformation matrix first two dont work 
# Q3 = np.zeros((4,4))
# disparity_sift=np.zeros((h,w),dtype=np.int64)

# sift = cv2.SIFT_create()
# kpl, descl = sift.detectAndCompute(prectL,None)
# kpr, descr = sift.detectAndCompute(prectR,None)
# r,c= prectR.shape
# # FLANN algorithm parameters
# FLANN_INDEX_KDTREE = 1
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks=50)


# flann = cv2.FlannBasedMatcher(index_params,search_params)
# matches = flann.knnMatch(descl,descr,k=2)
# good = []
# pts1 = []
# pts2 = []
# for i,(m,n) in enumerate(matches):
# # vary the constant in if statement for required matching
#     if m.distance < 0.8*n.distance:
#         good.append(m)
#         pts1.append(kpl[m.queryIdx].pt)
#         pts2.append(kpr[m.trainIdx].pt)
# # show correspondences in the two frames
# kp_img = cv2.drawMatches(prectL, kpl, prectR, kpr, good, None,matchColor=[0,255,0], flags=2)

# pts1 = np.array(pts1)
          
# pts2 = np.array(pts2)
#             # filter based on horizontal criteria
# # pts1,pts2 = filter_points(pts1,pts2,th=80)
#         # find the fundamental matrix
# pts1 = np.int32(pts1)
# pts2 = np.int32(pts2)

# b_size = 15
# for pt1,pt2 in zip(pts1,pts2):
#         j,i = map(int,(pt1[0],pt1[1]))
#         u,v = map(int,(pt2[0],pt2[1]))
#         # print(i,u,j,v)

#         disp_1 = j-u

#         disparity_sift[i-b_size:i+b_size,j-b_size:j+b_size]= (disparity_sift[i-b_size:i+b_size,j-b_size:j+b_size]+disp_1)/2
# plt.subplot(231)
# plt.imshow(disparity_bm,'plasma')
plt.subplot(325)
plt.imshow(disparity_sgbm,'plasma')
# plt.subplot(326)
# plt.imshow(disparity_sift,'plasma')
# plt.subplot(234)
# plt.imshow(prectL,'gray')
# plt.subplot(235)
# plt.imshow(prectL,'gray')
# plt.subplot(236)
# plt.imshow(prectL,'gray')

# cv2.mean

# cv2.imwrite(f"disparity_bm{img_no}.png",disparity_bm)
# cv2.imwrite(f"disparity_sgbm{img_no}.png",disparity_sgbm)
# # cv2.imwrite(f"disparity_sift{img_no}.png",disparity_sift)

# #Reproject points into 3D
# points_3D_bm = cv2.reprojectImageTo3D(disparity_bm, Q2)
points_3D_sgbm = cv2.reprojectImageTo3D(disparity_sgbm, Q2)
# points_3D_sift = cv2.reprojectImageTo3D(disparity_sift, Q2)
# #Get color points
colors = cv2.cvtColor(prectL, cv2.COLOR_BGR2RGB)
# #Get rid of points with value 0 (i.e no depth)

mask_map = np.ones(disparity_sgbm.shape[:2],dtype=np.bool)
# mask_map = mask_map[disparity_sgbm>disparity_sgbm.min]

# print(np.unique(disparity_sgbm))
# #Mask colors and points. 
# output_points_bm = points_3D_bm[mask_map]
output_points_sgbm = points_3D_sgbm[mask_map]
# output_points_sift = points_3D_sift[mask_map]
output_colors = colors[mask_map]
# #Define name for output file
# output_points = np.where(np.isinf(output_points_sgbm),0,output_points_sgbm)
# z_values  = output_points[:,2]

# index = np.argwhere(z_values>40000)
# print(output_points.shape)
# # output_points = output_points[index]
# # output_points= output_points.squeeze(1)
# # output_colors = output_colors[index]
# # output_colors = output_colors.squeeze(1)

# print(output_points.shape)



# output_file_sift = f'SIFT{str(img_no)}.ply'
output_file_sgbm = f'SGBM{str(img_no)}.ply'
# output_file_bm = f'BM{str(img_no)}.ply'
# # print(output_points)
# #Generate point cloud 
print (" Creating the output file... ")
# write_pointcloud(output_points_bm, output_colors, output_file_bm)
write_pointcloud(output_points_sgbm, output_colors, output_file_sgbm)
print ("\n Created the output file... \n")
plt.show()
# write_pointcloud(output_points_sift, output_colors, output_file_sift)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
# C= output_colors
# ax.scatter(output_points, c = C/255.0)
# plt.show()



