from pypylon import pylon
import cv2
import numpy as np

print(' d: data capture \n q: quit \n')

# directories where the images will be saved
j = 60
datadirL = './basler_L_data_2020/' #'./basler_L_data/'
datadirR = './basler_R_data_2020/' #'./basler_R_data/'

# Pypylon get camera by serial number
serial_number_1 = '22730679' #right
serial_number_2 = '22730681' #left
cams = []

for i in pylon.TlFactory.GetInstance().EnumerateDevices():
    cams.append(i)

print(cams[0].GetSerialNumber(), cams[1].GetSerialNumber())
camera_1 = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(cams[0]))
camera_2 = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(cams[1]))
camera_1.Open()
camera_2.Open()
print("[Cameras opened]")

# Grabing Continusely (video) with minimal delay
camera_1.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
camera_2.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()
print("[Grabbed Continuously (video) with minimal delay]")

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
print("[converted to opencv bgr format]")

print("[entering the loop]")
frame_count = 0
while camera_1.IsGrabbing() and camera_2.IsGrabbing():
    grabResult_1 = camera_1.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    grabResult_2 = camera_2.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult_1.GrabSucceeded() and grabResult_2.GrabSucceeded():
        # Access the image data
        image_1 = converter.Convert(grabResult_1)
        image_2 = converter.Convert(grabResult_2)

        img_1 = image_1.GetArray()
        img_2 = image_2.GetArray()

        # resolution of camera (1600,1200)
        # scaling it down by 3
        img_1 = cv2.resize(img_1, (1600,1200)) #right
        img_2 = cv2.resize(img_2, (1600,1200)) #left
        
        # Scaling it down only for display purposes
        cv2.namedWindow('right')#, cv2.WINDOW_GUI_NORMAL)
        cv2.namedWindow('left')#, cv2.WINDOW_GUI_NORMAL)
        cv2.imshow('right', cv2.resize(img_1, (640,480)))
        cv2.imshow('left', cv2.resize(img_2, (640,480)))
        
        # saving images into the respective folders
        if cv2.waitKey(1) & 0xff == ord('d'):
            cv2.imwrite(datadirR+str(j)+'_R_.png', img_1) #right
            cv2.imwrite(datadirL+str(j)+'_L_.png', img_2) #left
            print('data image ', j, ' saved')
            j+=1
        elif cv2.waitKey(1) & 0xff == ord('q'):
            break
        #endif

        frame_count += 1
        print(f"Frames count: ", frame_count)
    else:
        print("Image grabbing error: plug USB to some other port!")
    #endif
    grabResult_1.Release()
    grabResult_2.Release()
    
# Releasing the resource    
camera_1.StopGrabbing()
camera_2.StopGrabbing()

cv2.destroyAllWindows()