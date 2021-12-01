import cv2
import numpy as np

print(' d: data capture \n q: quit \n')
cap1 = cv2.VideoCapture(2)
cap2 = cv2.VideoCapture(3)
#cap1.set(cv2.CAP_PROP_FOCUS,3)
#cap2.set(cv2.CAP_PROP_FOCUS,3)
cap1.set(cv2.CAP_PROP_SETTINGS,1)
cap2.set(cv2.CAP_PROP_SETTINGS,1)

hori, vert = 1920, 1080

cap1.set(cv2.CAP_PROP_FRAME_WIDTH,hori)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT,vert)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH,hori)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT,vert)

i = 0
j = 30

datadirL = './21_1080p_L_data/'
datadirR = './21_1080p_R_data/'

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    chairL = frame1.copy()
    chairR = frame2.copy()
    y,count = 0,0
    while(count<40):
        chairR = cv2.line(chairR, (0,180+y), (1920,180+y), (255-3*count,0,0), 2)
        chairL = cv2.line(chairL, (0,180+y), (1920,180+y), (255-3*count,0,0), 2)
        y+=50
        count+=1

    # scaling imshow down to save memory
    cv2.imshow('right', cv2.resize(chairR, (640,360)))
    cv2.imshow('left', cv2.resize(chairL, (640,360)))

    if cv2.waitKey(1) & 0xff == ord('d'):
        cv2.imwrite(datadirR+str(j)+'_R_.png', frame2)
        cv2.imwrite(datadirL+str(j)+'_L_.png',frame1)
        print('data image ', j, ' saved')
        j+=1
    elif cv2.waitKey(1) & 0xff == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()

