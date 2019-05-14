
import cv2
 
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('test.mp4') #读取视频
 # 判断视频是否读取成功
if (cap.isOpened()== False):
  print("Error opening video stream or file")

#获取帧
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
    # 在每一帧上画矩形，frame帧,(四个坐标参数),（颜色）,宽度
    cv2.rectangle(frame, (int(200), int(300)), (int(400), int(500)), (255, 255, 255), 4)
    # 显示视频
    cv2.imshow('Frame',frame)
    
    # 刷新视频
    cv2.waitKey(10)
 
    # 按q退出
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
 
  # Break the loop
  else:
    break 