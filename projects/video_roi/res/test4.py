import cv2

drawing = False
ix, iy = -1, -1
tempFlag = False
def draw_circle(event, x, y, flags, param):
    global ix, iy, drawing, mode, cap, template, tempFlag
    if event == cv2.EVENT_LBUTTONDOWN:
        tempFlag = True
        drawing = True
        ix, iy = x, y                        #按下鼠标左键，用全局变量ix,iy记录下当前坐标点
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False                  #鼠标左键抬起，画出矩形框
            cv2.rectangle(frame, (ix, iy), (x, y), (0, 255, 0), 1)
            template = frame[iy:y, ix:x, :]  #截取框中的目标图像
            cap = cv2.VideoCapture(-1)       #打开摄像头
            cv2.imshow('img', frame)         #显示画框后的图像

cv2.namedWindow('img')
cv2.setMouseCallback('img', draw_circle)

cap = cv2.VideoCapture('test.mp4')
while (True):
    ret, frame = cap.read()
    cv2.imshow('Vedio', frame)
    if tempFlag == True:         #如果框出了目标，显示该目标
        cv2.imshow('temp', template)
    k = cv2.waitKey(33)
    if k == 27:                  #退出视频
        break
    elif k == 32:                #如果按下空格键
        while(1):
            cap.release()        #关掉摄像头
            imgCOPY = frame      #显示关闭摄像头前最后一张图像
            cv2.imshow('img', frame)
            k = cv2.waitKey(0)   #等待调用鼠标回调函数框出目标
            if k == 32:          #框完目标，再次按下空格键，摄像头捕捉的画面重新播放
                break