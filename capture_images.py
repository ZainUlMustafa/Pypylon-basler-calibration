from time import sleep
from pypylon import pylon
import cv2
import numpy as np
cv2.destroyAllWindows()

print(' d: data capture \n q: quit \n')

# directories where the images will be saved
j = 100
datadirL = './captured/left/'
datadirR = './captured/right/'

# Pypylon get camera by serial number
serial_number_1 = '22730679' #right
serial_number_2 = '22730681' #left
cams = []

selector = 0
selector = int(input("Both: 0\nRight: 1\nLeft: 2\nTest: 9\nInput:"))
if selector == 0:
    datadirL = './captured/both/left/'
    datadirR = './captured/both/right/'
#endif

if selector == 9:
    datadirL = './captured/test/left/'
    datadirR = './captured/test/right/'
#endif

for i in pylon.TlFactory.GetInstance().EnumerateDevices():
    cams.append(i)
#endif

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
auto = 0
auto = int(input("Auto capture (1 or 0):"))

# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (50, 50)
# fontScale
fontScale = 1
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2

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
        img_1_show = cv2.resize(img_1, (1600,1200)) #right
        img_2_show = cv2.resize(img_2, (1600,1200)) #left

        cv2.putText(img_1_show, f'{j} pictures captured', org, font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.putText(img_2_show, f'{j} pictures captured', org, font, fontScale, color, thickness, cv2.LINE_AA)
        
        # Scaling it down only for display purposes
        cv2.namedWindow('right')#, cv2.WINDOW_GUI_NORMAL)
        cv2.namedWindow('left')#, cv2.WINDOW_GUI_NORMAL)
        cv2.imshow('right', cv2.resize(img_1_show, (640*1,480*1)))
        cv2.imshow('left', cv2.resize(img_2_show, (640*1,480*1)))

        # saving images into the respective folders
        if auto == 1:
            if selector == 1 or selector == 0 or selector == 9:
                cv2.imwrite(datadirR+str(j)+'_R_.png', img_1) #right
            
            if selector == 2 or selector == 0 or selector == 9:
                cv2.imwrite(datadirL+str(j)+'_L_.png', img_2) #left
            
            print('data image ', j, ' saved')
            j+=1
            # sleep(2)
            
            cv2.waitKey(5000)
        else:
            if cv2.waitKey(1) & 0xff == ord('d'):
                if selector == 1 or selector == 0 or selector == 9:
                    cv2.imwrite(datadirR+str(j)+'_R_.png', img_1) #right
            
                if selector == 2 or selector == 0 or selector == 9:
                    cv2.imwrite(datadirL+str(j)+'_L_.png', img_2) #left
                
                print('data image ', j, ' saved')
                j+=1
            elif cv2.waitKey(1) & 0xff == ord('q'):
                break
            #endif
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