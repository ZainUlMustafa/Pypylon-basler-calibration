import cv2
import os
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import json
import stereo_setting as stset

# Rectifying the images
img_no = 101
imgL, imgR = cv2.imread(f"./test/left/{str(img_no)}_L_.png"), cv2.imread(f"./test/right/{str(img_no)}_R_.png")
print(imgL.shape[:2])
print('IMAGES LOADED')
print(100*'#')

vert, hori = imgL.shape[:2]
left_stereo_map, right_stereo_map, _ = stset.st_maps("./calibrators/calibParams/", (hori, vert))
print('MAPS COMPUTED')
print(100*'#')

rectL, rectR = stset.st_rectify(imgL, imgR, left_stereo_map, right_stereo_map)
#rectL, rectR = cv2.imread("./singlerun/rectL2.png"), cv2.imread("./singlerun/rectR2.png")
print('RECTIFIED')
print(100*'#')
grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)
rectified_pair = (grayR, grayL)

# Depth map function
'''
cv2.StereoSGBM_create(
    minDisparity=None, 
    numDisparities=None, 
    blockSize=None, 
    P1=None, 
    P2=None, 
    disp12MaxDiff=None, 
    preFilterCap=None, 
    uniquenessRatio=None, 
    speckleWindowSize=None, 
    speckleRange=None, 
    mode=None)
'''
BS = 7
MDS = 1
NOD = 192
UR = 10
SPWS = 100
SR = 32
DMD = 5
P1 = 8*3*BS**2
P2 = 32*3*BS**2 

def stereo_depth_map(rectified_pair):
    print ('BS='+str(BS)+' MDS='+str(MDS)+' NOD='+str(NOD)+' UR='+\
           str(UR)+' SPWS='+str(SPWS)+' SR='+str(SR))
    print (' DMD='+str(DMD)+' P1='+str(P1)+' P2='+str(P2))
    c, r = rectified_pair[0].shape
    disparity = np.zeros((c, r), np.uint8)
    sbm = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)
    sbm.setBlockSize(BS)
    sbm.setMinDisparity(MDS)
    sbm.setNumDisparities(NOD)
    sbm.setUniquenessRatio(UR)
    sbm.setSpeckleWindowSize(SPWS)
    sbm.setSpeckleRange(SR)
    sbm.setDisp12MaxDiff(DMD)
    sbm.setP1(8*3*BS**2)
    sbm.setP2(32*3*BS**2)

    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    #cv2.FindStereoCorrespondenceBM(dmLeft, dmRight, disparity, sbm)
    disparity = sbm.compute(dmLeft, dmRight)
    #disparity_visual = cv.CreateMat(c, r, cv.CV_8U)
    local_max = disparity.max()
    local_min = disparity.min()
    print ("MAX " + str(local_max))
    print ("MIN " + str(local_min))
    disparity_visual = (disparity-local_min)*(1.0/(local_max-local_min))
    local_max = disparity_visual.max()
    local_min = disparity_visual.min()
    print ("MAX " + str(local_max))
    print ("MIN " + str(local_min))
    #cv.Normalize(disparity, disparity_visual, 0, 255, cv.CV_MINMAX)
    #disparity_visual = np.array(disparity_visual)
    return disparity_visual

disparity = stereo_depth_map(rectified_pair)

# Set up and draw interface
# Draw left image and depth map
axcolor = 'lightgoldenrodyellow'
fig = plt.subplots(1,2)
plt.subplots_adjust(left=0.15, bottom=0.5)
plt.subplot(1,2,1)
dmObject = plt.imshow(rectified_pair[0], 'gray')

saveax = plt.axes([0.3, 0.38, 0.15, 0.04]) #stepX stepY width height
buttons = Button(saveax, 'Save settings', color=axcolor, hovercolor='0.975')


def save_map_settings( event ):
    buttons.label.set_text ("Saving...")
    print('Saving to file...') 
    result = json.dumps({'ImageNo':img_no,'blockSize':BS, 'minDisparity':MDS, 'numDisparities':NOD, \
             'uniquenessRatio':UR, 'speckleWindowSize':SPWS, 'speckleRange':SR, \
             'disp12MaxDiff':DMD, 'P1':P1, 'P2':P2},\
             sort_keys=True, indent=4, separators=(',',':'))
    fName = '3dmap_set.txt'
    f = open (str(fName), 'w') 
    f.write(result)
    f.close()
    buttons.label.set_text ("Save to file")
    print ('Settings saved to file '+fName)

buttons.on_clicked(save_map_settings)

loadax = plt.axes([0.5, 0.38, 0.15, 0.04]) #stepX stepY width height
buttonl = Button(loadax, 'Load settings', color=axcolor, hovercolor='0.975')
def load_map_settings( event ):
    global loading_settings, BS, MDS, NOD, UR, SPWS, SR, DMD, P1, P2
    loading_settings = 1
    fName = '3dmap_set.txt'
    print('Loading parameters from file...')
    buttonl.label.set_text ("Loading...")
    f=open(fName, 'r')
    data = json.load(f)
    sBS.set_val(data['blockSize'])
    sMDS.set_val(data['minDisparity'])
    sNOD.set_val(data['numDisparities'])
    sUR.set_val(data['uniquenessRatio'])
    sSPWS.set_val(data['speckleWindowSize'])
    sSR.set_val(data['speckleRange'])
    sDMD.set_val(data['disp12MaxDiff'])
    sP1.set_val(data['P1'])
    sP2.set_val(data['P2'])    
    f.close()
    buttonl.label.set_text ("Load settings")
    print ('Parameters loaded from file '+fName)
    print ('Redrawing depth map with loaded parameters...')
    loading_settings = 0
    update(0)
    print ('Done!')

buttonl.on_clicked(load_map_settings)


plt.subplot(1,2,2)
dmObject = plt.imshow(disparity, aspect='equal', cmap='gray')

# Draw interface for adjusting parameters
print('Start interface creation (it takes up to 30 seconds)...')

SWSaxe = plt.axes([0.15, 0.01, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height 
PFSaxe = plt.axes([0.15, 0.05, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height 
PFCaxe = plt.axes([0.15, 0.09, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height 
MDSaxe = plt.axes([0.15, 0.13, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height 
NODaxe = plt.axes([0.15, 0.17, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height 
TTHaxe = plt.axes([0.15, 0.21, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height 
URaxe = plt.axes([0.15, 0.25, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height
SRaxe = plt.axes([0.15, 0.29, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height
SPWSaxe = plt.axes([0.15, 0.33, 0.7, 0.025], facecolor=axcolor) #stepX stepY width height

sBS = Slider(SWSaxe, 'BlockSize', 5.0, 255.0, valinit=5)
sMDS = Slider(PFSaxe, 'MinDisp', -100.0, 100.0, valinit=5)
sNOD = Slider(PFCaxe, 'NumOfDisp', 16.0, 640.0, valinit=16)
sUR = Slider(MDSaxe, 'UnicRatio', 1.0, 20.0, valinit=2)
sSPWS = Slider(NODaxe, 'SpklWinSze', 0.0, 300.0, valinit=128)
sSR = Slider(TTHaxe, 'SpcklRng', 0.0, 40.0, valinit=10)
sDMD = Slider(URaxe, 'DispMaxDiff', 1.0, 20.0, valinit=10)
sP1 = Slider(SRaxe, 'P_1', 0.0, 5000.0, valinit=15)
sP2 = Slider(SPWSaxe, 'P_2', 0.0, 5000.0, valinit=15)

# Produce the colormap disparity image
def color_disparity_map(disparity):
    local_max = disparity.max()
    local_min = disparity.min()
    disparity_grayscale = (disparity-local_min)*(65535.0/(local_max-local_min))
    disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
    disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)

    norm_coeff = 255 / disparity.max()
    return disparity * norm_coeff / 255, disparity_color
#enddef

def updatedepthmap(event):
        print ('Rebuilding depth map')
        disparity = stereo_depth_map(rectified_pair)
        dmObject.set_data(disparity)
        print('Saving disp map!')
        disp, cdisp = color_disparity_map(disparity)
        cv2.imwrite('./disp.png', cdisp)
        print ('Redraw depth map')
        plt.draw()
uax= plt.axes([0.7, 0.38, 0.15, 0.04])
buttonu = Button(uax, 'Update Map', color=axcolor, hovercolor='0.975')

buttonu.on_clicked(updatedepthmap)


# Update depth map parameters and redraw
def update(val):
    global loading_settings, BS, MDS, NOD, UR, SPWS, SR, DMD, P1, P2
    BS = int(sBS.val/2)*2+1 #convert to ODD   
    MDS = int(sMDS.val)    
    NOD = int(sNOD.val/16)*16
    UR = int(sUR.val)  
    SPWS= int(sSPWS.val)
    SR = int(sSR.val)
    P1 = 8*3*BS**2#int(sP1.val)
    P2 = 32*3*BS**2#int(sP2.val)
    
    
# Connect update actions to control elements
sBS.on_changed(update)
sMDS.on_changed(update)
sNOD.on_changed(update)
sUR.on_changed(update)
sSPWS.on_changed(update)
sSR.on_changed(update)
sP1.on_changed(update)
sP2.on_changed(update)
sSPWS.on_changed(update)

print('Show interface to user')
plt.show()
