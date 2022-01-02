from pypylon import pylon
import cv2
import numpy as np

for i in pylon.TlFactory.GetInstance().EnumerateDevices():
    cams.append(i)
#endfor

for cam in cams:
    print(cam.GetSerialNumber())
#endfor