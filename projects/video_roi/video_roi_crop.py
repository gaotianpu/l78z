#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
加载视频，
    用户拖拽选定感兴趣区域，
    右键取消选中区域
    保存roi区域

鼠标事件定义：https://blog.51cto.com/devops2016/2084084
"""
import sys
import numpy as np
import cv2
import json
from skimage.measure import compare_ssim

def compare_image(grayA, imageB):
    '''比较2张图片的相似度'''
    # imageB = cv2.imread(path_image2)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    (score, diff) = compare_ssim(grayA, grayB, full=True)
    # print("SSIM: {}".format(score))
    return score

def run(video_id,video_file):
    '''对ROI区域抽帧裁剪''' 
    #定义窗口
    win_title = 'frame'
    cv2.namedWindow(win_title)
    cv2.moveWindow(win_title, 10, 100)

    #fps ？
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    fps_params = cv2.CAP_PROP_FPS
    if int(major_ver) < 3:
        fps_params = cv2.cv.CV_CAP_PROP_FPS  

    cap = cv2.VideoCapture(video_file)  # 文件名及格式 
    fps = cap.get(fps_params) 
    
    # 从人工选取的数据文件中加载要剪裁的区间
    sample_area = []
    selected_area = [] 
    with open("output/a/%s.json" % (video_id) , 'r') as f: 
        tmp = json.load(f) 
        selected_area = tmp['selected'] 
        for i,area in enumerate(selected_area):
            manual_img = "output/a/%s_%s.png" % (video_id,i)  
            # print(manual_img)
            imageA = cv2.imread(manual_img)
            grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY) 
            sample_area.append(grayA)
             
    
    frame = np.zeros((500, 500, 3), np.uint8)

    frame_index = 0
    while(True):
        #capture frame-by-frame
        ret, frame = cap.read()  
        if not ret:
            break 
        
        if frame_index % (fps*1) == 0:  # 每间隔5s抽帧
            for i,area in enumerate(selected_area):
                min_x = min(area[1][0],area[2][0])     
                min_y = min(area[1][1],area[2][1])
                width = abs(area[1][0] - area[2][0])
                height = abs(area[1][1] -area[2][1])
                cropped_img = frame[min_y:min_y+height, min_x:min_x+width] 
                score = compare_image(sample_area[i], cropped_img)
                if score>=0.3 and score<0.95:
                    print(i,frame_index,score) 
                    sample_img = "output/b/%s_%s_%s.png"%(video_id,i,frame_index) # media_id, selected_area_no,frame_index
                    cv2.imwrite(sample_img, cropped_img)  

        for area in selected_area: 
            cv2.rectangle(frame, tuple(area[1]), tuple(area[2]), (0, 255, 0), 1)

        # #display the resulting frame
        cv2.imshow(win_title, frame)
        if cv2.waitKey(25) & 0xFF == 27:  # 按esc键退出  # 0xFF == ord('q'):  # 按q键退出  
            break
        
        frame_index = frame_index + 1 

    #when everything done , release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_id = sys.argv[1]
    video_file = sys.argv[2]
     
    run(video_id,video_file)
