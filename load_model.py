import cv2
import numpy as np 
from pypylon import pylon
import stereo_setting as stset

################################################################################################

def draw_lines(image, hori):
    y,count = 0,0
    while(count<40):
        image = cv2.line(image, (0,50+y), (hori,50+y), (count*4,y,count*2), 2)
        y+=60
        count+=1

    return image
#enddef

################################################################################################

def filtered_disparity_map(rectL, rectR):
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    # Create StereoSGBM and prepare all parameters
    window_size = 3
    min_disp = 1
    num_disp = (16*13)+1-min_disp
    left_matcher = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = window_size,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32,
        disp12MaxDiff = 5,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2)

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

    return filteredImg
#enddef

################################################################################################

def disparity_map(rectL, rectR):
    sbm = cv2.StereoSGBM_create(numDisparities=16, blockSize=15)
    sbm.setBlockSize(5)
    sbm.setMinDisparity(47)
    sbm.setNumDisparities(320)
    sbm.setUniquenessRatio(3)
    sbm.setSpeckleWindowSize(219)
    sbm.setSpeckleRange(23)
    sbm.setDisp12MaxDiff(5)
    sbm.setP1(8*3*5**2)
    sbm.setP2(32*3*5**2)
    disparity_img = sbm.compute(rectL, rectR)
    norm_coeff = 255 / disparity_img.max()

    return disparity_img * norm_coeff / 255
#enddef

def color_disparity_map(rectL, rectR):
    stereoMatcher = cv2.StereoSGBM_create()
    stereoMatcher.setMinDisparity(0)
    stereoMatcher.setNumDisparities(16*42)
    stereoMatcher.setBlockSize(20)
    stereoMatcher.setSpeckleRange(16)
    stereoMatcher.setSpeckleWindowSize(45)

    disparity = stereoMatcher.compute(rectL, rectR)
    local_max = disparity.max()
    local_min = disparity.min()
    disparity_grayscale = (disparity-local_min)*(65535.0/(local_max-local_min))
    disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
    disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
    return disparity_color
#enddef

################################################################################################

def main():
    #############################################################################################
    # Pypylon get camera by serial number
    serial_number_1 = '22730679' #right
    serial_number_2 = '22730681' #left
    cams = []

    for i in pylon.TlFactory.GetInstance().EnumerateDevices():
        cams.append(i)
    #next

    print(cams[0].GetSerialNumber(), cams[1].GetSerialNumber())
    camera_1 = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(cams[0]))
    camera_2 = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(cams[1]))
    camera_1.Open()
    camera_2.Open()

    # Grabing Continusely (video) with minimal delay
    camera_1.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
    camera_2.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
    converter = pylon.ImageFormatConverter()

    # Converting to opencv bgr format
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    #############################################################################################
    # Resolution of camera (1600,1200)
    hori, vert = 1600, 1200
    left_stereo_map, right_stereo_map, Q = stset.st_maps("./basler_calib_params_2020/", (hori, vert))
    print('MAPS COMPUTED')
    print(100*'#')

    #############################################################################################
    while camera_1.IsGrabbing() and camera_2.IsGrabbing():
        grabResult_1 = camera_1.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        grabResult_2 = camera_2.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult_1.GrabSucceeded() and grabResult_2.GrabSucceeded():
            # Access the image data
            image_1 = converter.Convert(grabResult_1) #right
            image_2 = converter.Convert(grabResult_2) #left

            imgR = image_1.GetArray()
            imgL = image_2.GetArray()

            #############################################################################################
            # Rectification process
            #rectL = imgL
            #rectR = imgR
            rectL, rectR = stset.st_rectify(imgL, imgR, left_stereo_map, right_stereo_map)
            print('RECTIFIED')
            print(100*'#')

            #############################################################################################
            # Scaling it down by 3
            resize_scale = 3
            scaled = (int(hori/resize_scale), int(vert/resize_scale))

            # Displaying the images and saving them
            cv2.imshow('imgL', cv2.resize(draw_lines(imgL.copy(), hori), scaled))
            cv2.imshow('imgR', cv2.resize(draw_lines(imgR.copy(), hori), scaled))
            
            cv2.imshow('rectL', cv2.resize(draw_lines(rectL.copy(), hori), scaled))
            cv2.imshow('rectR', cv2.resize(draw_lines(rectR.copy(), hori), scaled))

            cv2.imwrite("./singlerun_2020/left3.png", imgL)
            cv2.imwrite("./singlerun_2020/right3.png", imgR)

            cv2.imwrite("./singlerun_2020/rectL3.png", rectL)
            cv2.imwrite("./singlerun_2020/rectR3.png", rectR)
            
            #############################################################################################
            # displaying the image
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
            #endif
        #endif

        grabResult_1.Release()
        grabResult_2.Release()
    #endwhile

    # Releasing the resource    
    camera_1.StopGrabbing()
    camera_2.StopGrabbing()

    cv2.destroyAllWindows()
#enddef

################################################################################################

if __name__ == "__main__":
    main()


