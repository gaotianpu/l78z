'''按下按键，视频暂停播放，再次按下，视频继续播放'''
import numpy as np
import cv2

cap = cv2.VideoCapture('test.mp4')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',-1, 20.0, (1920,1080))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        # frame = cv2.flip(frame,0)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)      #灰度化
        # write the flipped frame
        #out.write(frame)
        cv2.namedWindow('frame',0)
        cv2.imshow('frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break
    #waitkey()中的延迟时间需要根据程序定义，太少了捕捉不到按键
    if(cv2.waitKey(30)>=0): 
        cv2.waitKey(0)  
cap.release()
#out.release()
cv2.destroyAllWindows() 